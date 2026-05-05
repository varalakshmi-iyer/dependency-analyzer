"""
Dependency Version Change Analyzer using Vertex AI (Gemini)
Analyzes PR diffs to detect breaking changes between dependency versions.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


@dataclass
class DependencyChange:
    name: str
    old_version: str
    new_version: str
    file: str


@dataclass
class BreakingChange:
    dependency: str
    old_version: str
    new_version: str
    change_type: str        # METHOD_REMOVED | PROPERTY_RENAMED | API_CHANGED | BEHAVIOR_CHANGED | CONFIG_CHANGED | DEPRECATION | OTHER
    description: str
    severity: Severity
    affected_code: Optional[str] = None
    migration_hint: Optional[str] = None


@dataclass
class AnalysisResult:
    dependency_changes: list[DependencyChange] = field(default_factory=list)
    breaking_changes:   list[BreakingChange]   = field(default_factory=list)
    warnings:           list[str]              = field(default_factory=list)
    summary:            str                    = ""
    safe_to_merge:      bool                   = True

    @property
    def has_critical(self) -> bool:
        return any(b.severity == Severity.CRITICAL for b in self.breaking_changes)

    @property
    def has_breaking(self) -> bool:
        return bool(self.breaking_changes)


# ---------------------------------------------------------------------------
# Diff parser  (pip · npm · Maven · Gradle · pyproject.toml)
# ---------------------------------------------------------------------------

_DEPENDENCY_PATTERNS = [
    # requirements.txt   requests==2.28.0
    (r"^\-\s*(?P<name>[\w\-\.]+)\s*(?:==|>=|<=|~=|!=)\s*(?P<old_ver>[\d\.\w\-\+]+)",
     r"^\+\s*(?P<name>[\w\-\.]+)\s*(?:==|>=|<=|~=|!=)\s*(?P<new_ver>[\d\.\w\-\+]+)"),
    # pyproject.toml / setup.cfg   requests = "^2.28"
    (r'^\-\s*(?P<name>[\w\-\.]+)\s*=\s*["\^~]?(?P<old_ver>[\d\.\w\-\+]+)',
     r'^\+\s*(?P<name>[\w\-\.]+)\s*=\s*["\^~]?(?P<new_ver>[\d\.\w\-\+]+)'),
    # package.json   "axios": "^1.4.0"
    (r'^\-\s*"(?P<name>[\w\-\@\/\.]+)":\s*"[\^~]?(?P<old_ver>[\d\.\w\-\+]+)"',
     r'^\+\s*"(?P<name>[\w\-\@\/\.]+)":\s*"[\^~]?(?P<new_ver>[\d\.\w\-\+]+)"'),
    # pom.xml   <version>1.0.0</version>
    (r"^\-.*<version>(?P<old_ver>[\d\.\w\-]+)</version>",
     r"^\+.*<version>(?P<new_ver>[\d\.\w\-]+)</version>"),
    # build.gradle   implementation 'group:artifact:1.0.0'
    (r"""^\-\s*(?:implementation|api|compile|testImplementation)\s+['"](?P<name>[\w\.\-\:]+):(?P<old_ver>[\d\.\w\-]+)['"]""",
     r"""^\+\s*(?:implementation|api|compile|testImplementation)\s+['"](?P<name>[\w\.\-\:]+):(?P<new_ver>[\d\.\w\-]+)['"]"""),
]


def parse_dependency_changes(diff: str) -> list[DependencyChange]:
    """Extract dependency version bumps from a unified diff string."""
    changes: list[DependencyChange] = []
    current_file = "unknown"
    removed: list[str] = []
    added:   list[str] = []

    def _flush(file_name: str):
        for rem in removed:
            for add in added:
                for rem_pat, add_pat in _DEPENDENCY_PATTERNS:
                    m_rem = re.match(rem_pat, rem.strip(), re.IGNORECASE)
                    m_add = re.match(add_pat, add.strip(), re.IGNORECASE)
                    if not m_rem or not m_add:
                        continue
                    name_rem = m_rem.groupdict().get("name", "")
                    name_add = m_add.groupdict().get("name", "")
                    if name_rem and name_add and name_rem.lower() != name_add.lower():
                        continue
                    name    = name_rem or name_add or "unknown"
                    old_ver = m_rem.groupdict().get("old_ver", "")
                    new_ver = m_add.groupdict().get("new_ver", "")
                    if old_ver and new_ver and old_ver != new_ver:
                        changes.append(DependencyChange(name, old_ver, new_ver, file_name))

    for line in diff.splitlines():
        if line.startswith("diff --git") or line.startswith("--- ") or line.startswith("+++ "):
            _flush(current_file); removed.clear(); added.clear()
            if line.startswith("+++ "):
                current_file = line[4:].strip().lstrip("b/")
        elif line.startswith("@@"):
            _flush(current_file); removed.clear(); added.clear()
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line)

    _flush(current_file)

    # Deduplicate
    seen, unique = set(), []
    for c in changes:
        key = (c.name, c.old_version, c.new_version)
        if key not in seen:
            seen.add(key); unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Vertex AI client
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a senior software engineer specializing in library compatibility and migration.
Analyze the dependency version upgrade provided and identify breaking changes,
deprecated APIs, renamed methods/properties, or behavioral differences.

Focus ONLY on the dependency change — do NOT invent unrelated changes.

Respond with a JSON object (no markdown fences) with this exact shape:
{
  "breaking_changes": [
    {
      "change_type": "METHOD_REMOVED | PROPERTY_RENAMED | API_CHANGED | BEHAVIOR_CHANGED | CONFIG_CHANGED | DEPRECATION | OTHER",
      "description": "<clear description>",
      "severity": "CRITICAL | HIGH | MEDIUM | LOW | INFO",
      "affected_code": "<optional: affected usage pattern>",
      "migration_hint": "<optional: how to migrate>"
    }
  ],
  "warnings": ["<non-breaking concerns>"],
  "summary": "<one-paragraph plain-English summary>",
  "safe_to_merge": true | false
}

Severity guide:
  CRITICAL – runtime crash / data loss guaranteed
  HIGH     – very likely runtime errors or silent data corruption
  MEDIUM   – may break depending on usage; careful review needed
  LOW      – minor changes, easily handled
  INFO     – purely informational
""".strip()


class DependencyAnalyzer:
    """Analyzes dependency version changes in a PR diff using Vertex AI Gemini."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ):
        self.project_id       = project_id
        self.location         = location
        self.model_name       = model_name
        self.temperature      = temperature
        self.max_output_tokens = max_output_tokens
        self._model: Optional[GenerativeModel] = None

    # ── private helpers ────────────────────────────────────────────────────

    def _init_model(self) -> GenerativeModel:
        if self._model is None:
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
        return self._model

    def _extract_diff_context(self, diff: str, dep: DependencyChange, context_lines: int = 20) -> str:
        lines = diff.splitlines()
        matches = [
            i for i, line in enumerate(lines)
            if dep.name in line and (dep.old_version in line or dep.new_version in line)
        ]
        if not matches:
            return ""
        c = matches[0]
        return "\n".join(lines[max(0, c - context_lines): c + context_lines])

    def _build_prompt(self, dep: DependencyChange, context: str) -> str:
        return (
            f"Dependency : {dep.name}\n"
            f"Old version: {dep.old_version}\n"
            f"New version: {dep.new_version}\n"
            f"File       : {dep.file}\n\n"
            f"Relevant diff context:\n{context or '(none)'}\n\n"
            "Analyze the upgrade and return JSON only."
        )

    def _call_vertex(self, prompt: str) -> dict:
        model = self._init_model()
        config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        response = model.generate_content(prompt, generation_config=config)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    def _parse_response(self, data: dict, dep: DependencyChange):
        breaking = []
        for item in data.get("breaking_changes", []):
            try:
                sev = Severity(item.get("severity", "MEDIUM").upper())
            except ValueError:
                sev = Severity.MEDIUM
            breaking.append(BreakingChange(
                dependency=dep.name,
                old_version=dep.old_version,
                new_version=dep.new_version,
                change_type=item.get("change_type", "OTHER"),
                description=item.get("description", ""),
                severity=sev,
                affected_code=item.get("affected_code"),
                migration_hint=item.get("migration_hint"),
            ))
        return (
            breaking,
            data.get("warnings", []),
            data.get("summary", ""),
            bool(data.get("safe_to_merge", True)),
        )

    # ── public API ─────────────────────────────────────────────────────────

    def analyze_diff(self, diff: str) -> AnalysisResult:
        """Full pipeline: parse diff → detect deps → call Vertex AI per change."""
        result = AnalysisResult()
        result.dependency_changes = parse_dependency_changes(diff)

        if not result.dependency_changes:
            result.summary = "No dependency version changes detected."
            return result

        logger.info("Found %d dependency change(s). Analyzing...", len(result.dependency_changes))
        summaries, overall_safe = [], True

        for dep in result.dependency_changes:
            logger.info("Analyzing: %s  %s → %s", dep.name, dep.old_version, dep.new_version)
            try:
                context = self._extract_diff_context(diff, dep)
                prompt  = self._build_prompt(dep, context)
                data    = self._call_vertex(prompt)
                breaking, warnings, summary, safe = self._parse_response(data, dep)

                result.breaking_changes.extend(breaking)
                result.warnings.extend(warnings)
                summaries.append(f"[{dep.name} {dep.old_version}→{dep.new_version}] {summary}")
                if not safe:
                    overall_safe = False

            except Exception as exc:
                msg = f"Failed to analyze {dep.name}: {exc}"
                logger.error(msg)
                result.warnings.append(msg)

        result.summary       = "\n\n".join(summaries) or "Analysis complete."
        result.safe_to_merge = overall_safe and not result.has_critical
        return result