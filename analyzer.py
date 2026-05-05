"""
Dependency Version Change Analyzer using Vertex AI (Gemini)
Analyzes PR diffs to detect breaking changes between dependency versions.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & internal data models
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


class ChangeType(str, Enum):
    METHOD_REMOVED   = "METHOD_REMOVED"
    PROPERTY_RENAMED = "PROPERTY_RENAMED"
    API_CHANGED      = "API_CHANGED"
    BEHAVIOR_CHANGED = "BEHAVIOR_CHANGED"
    CONFIG_CHANGED   = "CONFIG_CHANGED"
    DEPRECATION      = "DEPRECATION"
    OTHER            = "OTHER"


@dataclass
class DependencyChange:
    name:        str
    old_version: str
    new_version: str
    file:        str


@dataclass
class BreakingChange:
    dependency:     str
    old_version:    str
    new_version:    str
    change_type:    str
    description:    str
    severity:       Severity
    affected_code:  Optional[str] = None
    migration_hint: Optional[str] = None


@dataclass
class AnalysisResult:
    dependency_changes: List[DependencyChange] = field(default_factory=list)
    breaking_changes:   List[BreakingChange]   = field(default_factory=list)
    warnings:           List[str]              = field(default_factory=list)
    summary:            str                    = ""
    safe_to_merge:      bool                   = True

    @property
    def has_critical(self) -> bool:
        return any(b.severity == Severity.CRITICAL for b in self.breaking_changes)

    @property
    def has_breaking(self) -> bool:
        return bool(self.breaking_changes)


# ---------------------------------------------------------------------------
# Pydantic schema — enforced on Gemini output via response_schema
# ---------------------------------------------------------------------------

class BreakingChangeSchema(BaseModel):
    change_type:    ChangeType
    description:    str
    severity:       Severity
    affected_code:  Optional[str] = None
    migration_hint: Optional[str] = None


class AnalysisResponseSchema(BaseModel):
    breaking_changes: List[BreakingChangeSchema] = Field(default_factory=list)
    warnings:         List[str]                  = Field(default_factory=list)
    summary:          str
    safe_to_merge:    bool


# ---------------------------------------------------------------------------
# Diff parser  (pip · npm · Maven · Gradle · pyproject.toml · libs.versions.toml)
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
    # libs.versions.toml   springBoot = "2.7.14"
    (r'^\-\s*(?P<name>[\w\-]+)\s*=\s*"(?P<old_ver>[\d\.\w\-\+]+)"',
     r'^\+\s*(?P<name>[\w\-]+)\s*=\s*"(?P<new_ver>[\d\.\w\-\+]+)"'),
]


def parse_dependency_changes(diff: str) -> List[DependencyChange]:
    """Extract dependency version bumps from a unified diff string."""
    changes: List[DependencyChange] = []
    current_file = "unknown"
    removed: List[str] = []
    added:   List[str] = []

    def _flush(file_name: str) -> None:
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
                path = line[4:].strip()
                current_file = path.removeprefix("b/")    # fix for "uild.gradle" bug
        elif line.startswith("@@"):
            _flush(current_file); removed.clear(); added.clear()
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line)

    _flush(current_file)

    # Deduplicate
    seen:   set                    = set()
    unique: List[DependencyChange] = []
    for c in changes:
        key = (c.name, c.old_version, c.new_version)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a senior software engineer specializing in library compatibility and migration.
Analyze the dependency version upgrade provided and identify breaking changes,
deprecated APIs, renamed methods/properties, or behavioral differences.

Focus ONLY on the dependency provided — do NOT invent unrelated changes.

Severity guide:
  CRITICAL – runtime crash / data loss guaranteed
  HIGH     – very likely runtime errors or silent data corruption
  MEDIUM   – may break depending on usage; careful review needed
  LOW      – minor changes, easily handled
  INFO     – purely informational
""".strip()


def _build_prompt(dep: DependencyChange, context: str) -> str:
    return (
        f"Dependency : {dep.name}\n"
        f"Old version: {dep.old_version}\n"
        f"New version: {dep.new_version}\n"
        f"File       : {dep.file}\n\n"
        f"Relevant diff context:\n{context or '(none)'}\n\n"
        "Analyze this upgrade and return the result."
    )


def _build_short_prompt(dep: DependencyChange) -> str:
    return (
        f"Dependency : {dep.name}\n"
        f"Old version: {dep.old_version}\n"
        f"New version: {dep.new_version}\n\n"
        "Based on your knowledge of this library's changelog, "
        "analyze this version upgrade. "
        "Focus only on the most impactful breaking changes — keep descriptions concise."
    )


# ---------------------------------------------------------------------------
# Vertex AI analyzer
# ---------------------------------------------------------------------------

class DependencyAnalyzer:
    """Analyzes dependency version changes in a PR diff using Vertex AI Gemini."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ):
        self.project_id        = project_id
        self.location          = location
        self.model_name        = model_name
        self.temperature       = temperature
        self.max_output_tokens = max_output_tokens
        self._model: Optional[GenerativeModel] = None

    # ── private ───────────────────────────────────────────────────────────

    def _init_model(self) -> GenerativeModel:
        if self._model is None:
            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(
                model_name=self.model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
            logger.info("Initialized Vertex AI model: %s", self.model_name)
        return self._model

    def _generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "breaking_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "change_type":    {"type": "string", "enum": [e.value for e in ChangeType]},
                                "description":    {"type": "string"},
                                "severity":       {"type": "string", "enum": [e.value for e in Severity]},
                                "affected_code":  {"type": "string"},
                                "migration_hint": {"type": "string"},
                            },
                            "required": ["change_type", "description", "severity"],
                        },
                    },
                    "warnings":      {"type": "array", "items": {"type": "string"}},
                    "summary":       {"type": "string"},
                    "safe_to_merge": {"type": "boolean"},
                },
                "required": ["breaking_changes", "warnings", "summary", "safe_to_merge"],
            },
        )

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

    def _repair_json(self, raw: str) -> str:
        """
        Best-effort repair of a truncated JSON string.
        Strips the incomplete trailing field and closes all open brackets.
        """
        logger.debug("Attempting JSON repair on: ...%s", raw[-200:])

        raw = re.sub(r',?\s*"[^"]*":\s*"[^"]*$', '', raw)  # incomplete string value
        raw = re.sub(r',?\s*"[^"]*":\s*$',        '', raw)  # key with no value
        raw = re.sub(r',?\s*"[^"]*$',             '', raw)  # incomplete key
        raw = raw.rstrip().rstrip(",")

        # Close any open arrays and objects
        open_brackets = raw.count("[") - raw.count("]")
        open_braces   = raw.count("{") - raw.count("}")
        raw += "]" * open_brackets
        raw += "}" * open_braces

        logger.debug("Repaired JSON tail: ...%s", raw[-200:])
        return raw

    def _safe_parse_response(self, raw: str, dep: DependencyChange) -> AnalysisResponseSchema:
        """
        Safely parse Gemini's response with repair fallback.
        Handles truncation and malformed JSON without crashing.
        """
        raw = raw.strip()

        if not raw or not raw.startswith("{"):
            logger.warning("%s: Response empty or not JSON — falling back.", dep.name)
            return self._fallback_response(dep)

        if not raw.endswith("}"):
            logger.warning("%s: Response appears truncated — attempting repair.", dep.name)
            raw = self._repair_json(raw)

        try:
            return AnalysisResponseSchema.model_validate_json(raw)
        except Exception as e:
            logger.warning("%s: Initial parse failed (%s) — attempting repair.", dep.name, e)
            try:
                return AnalysisResponseSchema.model_validate_json(self._repair_json(raw))
            except Exception as e2:
                logger.error("%s: Repair also failed (%s) — falling back.", dep.name, e2)
                return self._fallback_response(dep)

    def _fallback_response(self, dep: DependencyChange) -> AnalysisResponseSchema:
        """Conservative fallback when all parsing attempts fail."""
        return AnalysisResponseSchema(
            breaking_changes=[],
            warnings=[f"Could not analyze {dep.name} — manual review recommended."],
            summary=f"Analysis of {dep.name} {dep.old_version}→{dep.new_version} failed. Review manually.",
            safe_to_merge=False,
        )

    def _call_vertex(self, prompt: str, dep: DependencyChange) -> AnalysisResponseSchema:
        """Call Gemini with schema enforcement and truncation handling."""
        model    = self._init_model()
        response = model.generate_content(prompt, generation_config=self._generation_config())

        logger.debug("%s: Raw response: %s", dep.name, response.text)

        candidate     = response.candidates[0]
        finish_reason = candidate.finish_reason.name
        logger.debug("%s: finish_reason = %s", dep.name, finish_reason)

        if finish_reason == "MAX_TOKENS":
            logger.warning("%s: Hit MAX_TOKENS — retrying with short prompt.", dep.name)
            return self._call_vertex_short(dep)

        return self._safe_parse_response(response.text, dep)

    def _call_vertex_short(self, dep: DependencyChange) -> AnalysisResponseSchema:
        """Retry without diff context when full prompt exceeds token limit."""
        logger.info("%s: Retrying with knowledge-only prompt.", dep.name)
        model    = self._init_model()
        response = model.generate_content(
            _build_short_prompt(dep),
            generation_config=self._generation_config(),
        )

        candidate     = response.candidates[0]
        finish_reason = candidate.finish_reason.name

        if finish_reason == "MAX_TOKENS":
            logger.error("%s: Still truncated on short prompt — falling back.", dep.name)
            return self._fallback_response(dep)

        return self._safe_parse_response(response.text, dep)

    def _parse_response(
        self,
        data: AnalysisResponseSchema,
        dep: DependencyChange,
    ) -> Tuple[List[BreakingChange], List[str], str, bool]:
        breaking = [
            BreakingChange(
                dependency=dep.name,
                old_version=dep.old_version,
                new_version=dep.new_version,
                change_type=item.change_type.value,
                description=item.description,
                severity=Severity(item.severity.value),
                affected_code=item.affected_code,
                migration_hint=item.migration_hint,
            )
            for item in data.breaking_changes
        ]
        return breaking, data.warnings, data.summary, data.safe_to_merge

    # ── public ────────────────────────────────────────────────────────────

    def analyze_diff(self, diff: str) -> AnalysisResult:
        """Full pipeline: parse diff → detect deps → call Vertex AI per change."""
        result = AnalysisResult()
        result.dependency_changes = parse_dependency_changes(diff)

        if not result.dependency_changes:
            result.summary = "No dependency version changes detected."
            return result

        logger.info("Found %d dependency change(s). Analyzing...", len(result.dependency_changes))
        summaries:    List[str] = []
        overall_safe: bool      = True

        for dep in result.dependency_changes:
            logger.info("Analyzing: %s  %s → %s", dep.name, dep.old_version, dep.new_version)
            try:
                context = self._extract_diff_context(diff, dep)
                prompt  = _build_prompt(dep, context)
                data    = self._call_vertex(prompt, dep)
            except Exception as exc:
                logger.error(
                    "Vertex call failed for %s: %s: %s",
                    dep.name, type(exc).__name__, exc,
                )
                data = self._fallback_response(dep)

            breaking, warnings, summary, safe = self._parse_response(data, dep)
            result.breaking_changes.extend(breaking)
            result.warnings.extend(warnings)
            summaries.append(f"[{dep.name} {dep.old_version}→{dep.new_version}] {summary}")
            if not safe:
                overall_safe = False

        result.summary       = "\n\n".join(summaries) or "Analysis complete."
        result.safe_to_merge = overall_safe and not result.has_critical
        return result