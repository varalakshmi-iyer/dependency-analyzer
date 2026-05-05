"""
CLI for the Dependency Version Change Analyzer.

Usage:
  python cli.py --project my-gcp-project --diff pr.diff
  git diff main...HEAD | python cli.py --project my-gcp-project
  python cli.py --project my-gcp-project --pr-url https://github.com/org/repo/pull/42
  python cli.py --project my-gcp-project --diff pr.diff --output json --fail-on HIGH
"""

import argparse
import json
import logging
import os
import sys

from analyzer import AnalysisResult, BreakingChange, DependencyChange, DependencyAnalyzer, Severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("dep-analyzer")

# ANSI colours
_COLOUR = {
    Severity.CRITICAL: "\033[91m",
    Severity.HIGH:     "\033[31m",
    Severity.MEDIUM:   "\033[33m",
    Severity.LOW:      "\033[36m",
    Severity.INFO:     "\033[37m",
}
_RESET = "\033[0m"; _BOLD = "\033[1m"; _GREEN = "\033[32m"; _RED = "\033[91m"

def _c(text, colour, use_colour):
    return f"{colour}{text}{_RESET}" if use_colour else text


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_text(result: AnalysisResult, use_colour: bool = True) -> str:
    sep   = "─" * 70
    lines = [sep, _c("  DEPENDENCY VERSION CHANGE ANALYSIS", _BOLD, use_colour), sep]

    if result.dependency_changes:
        lines.append(f"\n{_c('Detected Dependency Upgrades', _BOLD, use_colour)} ({len(result.dependency_changes)})")
        for d in result.dependency_changes:
            lines.append(f"  • {d.name:40s}  {d.old_version:15s} → {d.new_version}  ({d.file})")
    else:
        lines.append("\n  No dependency version changes found.")

    if result.breaking_changes:
        lines.append(f"\n{_c('Breaking / Notable Changes', _BOLD, use_colour)} ({len(result.breaking_changes)})")
        for bc in result.breaking_changes:
            col   = _COLOUR.get(bc.severity, "")
            label = _c(f"[{bc.severity.value}]", col, use_colour)
            lines.append(f"\n  {label} {_c(bc.dependency, _BOLD, use_colour)}  {bc.old_version} → {bc.new_version}")
            lines.append(f"    Type   : {bc.change_type}")
            lines.append(f"    Detail : {bc.description}")
            if bc.affected_code:  lines.append(f"    Affects: {bc.affected_code}")
            if bc.migration_hint: lines.append(f"    Fix    : {bc.migration_hint}")
    else:
        lines.append(f"\n  {_c('No breaking changes detected.', _GREEN, use_colour)}")

    if result.warnings:
        lines.append(f"\n{_c('Warnings', _BOLD, use_colour)}")
        for w in result.warnings:
            lines.append(f"  ⚠  {w}")

    lines.append(f"\n{_c('Summary', _BOLD, use_colour)}")
    for para in result.summary.split("\n\n"):
        lines.append(f"  {para.strip()}")

    lines.append(f"\n{sep}")
    verdict = (_c("✔  SAFE TO MERGE", _GREEN, use_colour) if result.safe_to_merge
               else _c("✘  REVIEW REQUIRED — do NOT merge without addressing issues above", _RED, use_colour))
    lines += [f"  {verdict}", sep]
    return "\n".join(lines)


def format_json(result: AnalysisResult) -> str:
    return json.dumps({
        "safe_to_merge":      result.safe_to_merge,
        "summary":            result.summary,
        "dependency_changes": [vars(d) for d in result.dependency_changes],
        "breaking_changes": [
            {**{k: v for k, v in vars(b).items() if k != "severity"},
             "severity": b.severity.value}
            for b in result.breaking_changes
        ],
        "warnings": result.warnings,
    }, indent=2)


# ---------------------------------------------------------------------------
# Diff fetchers
# ---------------------------------------------------------------------------

def _from_file(path: str) -> str:
    from pathlib import Path
    return Path(path).read_text(encoding="utf-8")

def _from_stdin() -> str:
    return "" if sys.stdin.isatty() else sys.stdin.read()

def _from_github_pr(pr_url: str) -> str:
    import urllib.request
    token  = os.environ.get("GITHUB_TOKEN")
    parts  = pr_url.rstrip("/").split("/")
    idx    = parts.index("pull")
    owner, repo, pr_num = parts[idx - 2], parts[idx - 1], parts[idx + 1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"
    req = urllib.request.Request(api_url, headers={
        "Accept": "application/vnd.github.v3.diff",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SEV_ORDER = [Severity.INFO, Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

def _should_fail(result: AnalysisResult, fail_on: str | None) -> bool:
    if not fail_on:
        return False
    threshold = _SEV_ORDER.index(Severity(fail_on))
    return any(_SEV_ORDER.index(bc.severity) >= threshold for bc in result.breaking_changes)


def main():
    p = argparse.ArgumentParser(description="Analyze PR diff for dependency breaking changes via Vertex AI.")
    p.add_argument("--project",  required=True, help="GCP project ID")
    p.add_argument("--location", default="us-central1")
    p.add_argument("--model",    default="gemini-1.5-pro")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--diff",   metavar="FILE", help="Path to unified diff file")
    src.add_argument("--pr-url", metavar="URL",  help="GitHub PR URL (needs GITHUB_TOKEN)")

    p.add_argument("--output",   choices=["text", "json"], default="text")
    p.add_argument("--fail-on",  choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--verbose",  action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.diff:
        diff = _from_file(args.diff)
    elif args.pr_url:
        diff = _from_github_pr(args.pr_url)
    else:
        diff = _from_stdin()

    if not diff.strip():
        logger.error("Empty diff. Nothing to analyze.")
        sys.exit(2)

    analyzer = DependencyAnalyzer(
        project_id=args.project,
        location=args.location,
        model_name=args.model,
        temperature=args.temperature,
    )
    result = analyzer.analyze_diff(diff)

    use_colour = args.output == "text" and sys.stdout.isatty()
    print(format_json(result) if args.output == "json" else format_text(result, use_colour))

    if _should_fail(result, args.fail_on):
        logger.error("Build failed: breaking changes at or above --fail-on=%s", args.fail_on)
        sys.exit(1)


if __name__ == "__main__":
    main()