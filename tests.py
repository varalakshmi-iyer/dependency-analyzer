import json
from unittest.mock import MagicMock, patch

import pytest

from analyzer import (
    AnalysisResult, BreakingChange, DependencyAnalyzer,
    DependencyChange, Severity, parse_dependency_changes,
)
from cli import _should_fail, format_json, format_text


# ── parse_dependency_changes ──────────────────────────────────────────────

class TestParseDependencyChanges:

    def test_requirements_txt(self):
        diff = (
            "diff --git a/requirements.txt b/requirements.txt\n"
            "--- a/requirements.txt\n+++ b/requirements.txt\n"
            "@@ -1 +1 @@\n-requests==2.28.0\n+requests==2.31.0\n"
        )
        changes = parse_dependency_changes(diff)
        assert len(changes) == 1
        c = changes[0]
        assert c.name == "requests" and c.old_version == "2.28.0" and c.new_version == "2.31.0"

    def test_package_json(self):
        diff = (
            'diff --git a/package.json b/package.json\n--- a/package.json\n+++ b/package.json\n'
            '@@ -3 +3 @@\n-    "axios": "^1.4.0",\n+    "axios": "^1.6.0",\n'
        )
        changes = parse_dependency_changes(diff)
        assert any(c.name == "axios" and c.old_version == "1.4.0" and c.new_version == "1.6.0" for c in changes)

    def test_no_changes(self):
        diff = "diff --git a/main.py b/main.py\n--- a/main.py\n+++ b/main.py\n-print('a')\n+print('b')\n"
        assert parse_dependency_changes(diff) == []

    def test_deduplication(self):
        diff = (
            "diff --git a/requirements.txt b/requirements.txt\n--- a/req\n+++ b/req\n"
            "@@ -1 +1 @@\n-flask==2.2.0\n+flask==3.0.0\n"
            "@@ -5 +5 @@\n-flask==2.2.0\n+flask==3.0.0\n"
        )
        assert [c.name for c in parse_dependency_changes(diff)].count("flask") == 1

    def test_multiple_deps(self):
        diff = (
            "diff --git a/requirements.txt b/requirements.txt\n--- a/r\n+++ b/r\n"
            "@@ -1 +1 @@\n-requests==2.28.0\n+requests==2.31.0\n"
            "-boto3==1.26.0\n+boto3==1.34.0\n"
        )
        names = {c.name for c in parse_dependency_changes(diff)}
        assert "requests" in names and "boto3" in names


# ── AnalysisResult ────────────────────────────────────────────────────────

class TestAnalysisResult:

    def _bc(self, sev): return BreakingChange("lib", "1", "2", "API_CHANGED", "x", sev)

    def test_has_critical(self):
        assert AnalysisResult(breaking_changes=[self._bc(Severity.CRITICAL)]).has_critical
        assert not AnalysisResult(breaking_changes=[self._bc(Severity.HIGH)]).has_critical

    def test_has_breaking(self):
        r = AnalysisResult()
        assert not r.has_breaking
        r.breaking_changes.append(self._bc(Severity.LOW))
        assert r.has_breaking


# ── DependencyAnalyzer._parse_response ───────────────────────────────────

class TestParseResponse:

    def setup_method(self):
        self.a   = DependencyAnalyzer.__new__(DependencyAnalyzer)
        self.dep = DependencyChange("requests", "2.28.0", "2.31.0", "requirements.txt")

    def test_parses_correctly(self):
        data = {
            "breaking_changes": [{"change_type": "METHOD_REMOVED", "description": "x", "severity": "HIGH"}],
            "warnings": ["w"], "summary": "s", "safe_to_merge": False,
        }
        breaking, warnings, summary, safe = self.a._parse_response(data, self.dep)
        assert breaking[0].severity == Severity.HIGH and not safe

    def test_unknown_severity_defaults_medium(self):
        data = {"breaking_changes": [{"change_type": "OTHER", "description": "x", "severity": "BOGUS"}],
                "warnings": [], "summary": "", "safe_to_merge": True}
        breaking, *_ = self.a._parse_response(data, self.dep)
        assert breaking[0].severity == Severity.MEDIUM

    def test_empty(self):
        data = {"breaking_changes": [], "warnings": [], "summary": "ok", "safe_to_merge": True}
        breaking, _, summary, safe = self.a._parse_response(data, self.dep)
        assert breaking == [] and safe and summary == "ok"


# ── DependencyAnalyzer.analyze_diff (mocked Vertex) ──────────────────────

SAMPLE_DIFF = (
    "diff --git a/requirements.txt b/requirements.txt\n--- a/r\n+++ b/r\n"
    "@@ -1 +1 @@\n-requests==2.28.0\n+requests==2.31.0\n"
)

def _mock_model(payload: dict):
    m = MagicMock()
    m.generate_content.return_value.text = json.dumps(payload)
    return m

class TestAnalyzeDiff:

    def _analyzer(self): return DependencyAnalyzer(project_id="test")

    def test_no_changes(self):
        r = self._analyzer().analyze_diff("diff --git a/README.md b/README.md\n-a\n+b")
        assert r.dependency_changes == [] and r.safe_to_merge

    @patch("analyzer.vertexai.init")
    @patch("analyzer.GenerativeModel")
    def test_detects_and_analyzes(self, mock_cls, _):
        mock_cls.return_value = _mock_model({
            "breaking_changes": [{"change_type": "BEHAVIOR_CHANGED", "description": "x", "severity": "MEDIUM"}],
            "warnings": [], "summary": "ok", "safe_to_merge": True,
        })
        r = self._analyzer().analyze_diff(SAMPLE_DIFF)
        assert len(r.dependency_changes) == 1 and r.breaking_changes[0].severity == Severity.MEDIUM

    @patch("analyzer.vertexai.init")
    @patch("analyzer.GenerativeModel")
    def test_critical_sets_unsafe(self, mock_cls, _):
        mock_cls.return_value = _mock_model({
            "breaking_changes": [{"change_type": "API_CHANGED", "description": "x", "severity": "CRITICAL"}],
            "warnings": [], "summary": "bad", "safe_to_merge": False,
        })
        r = self._analyzer().analyze_diff(SAMPLE_DIFF)
        assert r.has_critical and not r.safe_to_merge

    @patch("analyzer.vertexai.init")
    @patch("analyzer.GenerativeModel")
    def test_vertex_error_adds_warning(self, mock_cls, _):
        m = MagicMock(); m.generate_content.side_effect = RuntimeError("quota")
        mock_cls.return_value = m
        r = self._analyzer().analyze_diff(SAMPLE_DIFF)
        assert any("Failed to analyze" in w for w in r.warnings)


# ── _should_fail ──────────────────────────────────────────────────────────

class TestShouldFail:

    def _r(self, sev):
        r = AnalysisResult()
        r.breaking_changes = [BreakingChange("lib", "1", "2", "X", "d", sev)]
        return r

    def test_no_fail_on(self):         assert not _should_fail(self._r(Severity.CRITICAL), None)
    def test_critical_on_critical(self): assert _should_fail(self._r(Severity.CRITICAL), "CRITICAL")
    def test_high_threshold_medium(self): assert not _should_fail(self._r(Severity.MEDIUM), "HIGH")
    def test_medium_threshold_high(self): assert _should_fail(self._r(Severity.HIGH), "MEDIUM")
    def test_empty_result(self):        assert not _should_fail(AnalysisResult(), "CRITICAL")