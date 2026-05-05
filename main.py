"""
main.py — Cloud Run webhook handler
Receives GitHub PR webhook events and triggers dependency analysis.
"""

import os
import hmac
import hashlib
import logging
import urllib.request
import urllib.error
from typing import Optional

from flask import Flask, request, jsonify

from analyzer import DependencyAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GCP_PROJECT       = os.environ["GCP_PROJECT"]
GCP_LOCATION      = os.environ.get("GCP_LOCATION", "us-central1")
GITHUB_TOKEN      = os.environ["GITHUB_TOKEN"]
WEBHOOK_SECRET    = os.environ.get("GITHUB_WEBHOOK_SECRET", "")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _github_request(url: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    import json
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(body).encode() if body else None
    if data:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github.v3.diff",   # returns raw diff
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def post_pr_comment(owner: str, repo: str, pr_number: int, body: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    _github_request(url, method="POST", body={"body": body})
    logger.info("Posted comment to PR #%d", pr_number)


def format_comment(result) -> str:
    icon = "✅" if result.safe_to_merge else "🚨"
    verdict = "SAFE TO MERGE" if result.safe_to_merge else "REVIEW REQUIRED"
    lines = [f"{icon} **Dependency Analysis: {verdict}**\n"]

    if result.dependency_changes:
        lines.append("| Package | Old | New | File |")
        lines.append("|---|---|---|---|")
        for d in result.dependency_changes:
            lines.append(f"| `{d.name}` | {d.old_version} | {d.new_version} | `{d.file}` |")
        lines.append("")

    if result.breaking_changes:
        lines.append("### Breaking Changes")
        for bc in result.breaking_changes:
            icon_sev = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(bc.severity.value, "⚪")
            lines.append(f"\n{icon_sev} **[{bc.severity.value}] {bc.dependency}** `{bc.old_version} → {bc.new_version}`")
            lines.append(f"- **Type**: {bc.change_type}")
            lines.append(f"- **Detail**: {bc.description}")
            if bc.affected_code:  lines.append(f"- **Affects**: `{bc.affected_code}`")
            if bc.migration_hint: lines.append(f"- **Fix**: {bc.migration_hint}")

    if result.warnings:
        lines.append("\n### Warnings")
        for w in result.warnings:
            lines.append(f"- ⚠️ {w}")

    lines.append(f"\n### Summary\n{result.summary}")
    lines.append("\n---\n_Powered by Vertex AI Gemini_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------

def _verify_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        logger.warning("No webhook secret set — skipping signature verification.")
        return True
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------

@app.route("/webhook", methods=["POST"])
def webhook():
    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_signature(request.data, signature):
        logger.warning("Invalid webhook signature — rejecting request.")
        return jsonify({"error": "Invalid signature"}), 401

    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return jsonify({"status": "ignored", "event": event}), 200

    payload = request.json
    action  = payload.get("action", "")

    # Only analyze on open/reopen/new commits
    if action not in ("opened", "synchronize", "reopened"):
        return jsonify({"status": "ignored", "action": action}), 200

    pr        = payload["pull_request"]
    pr_number = pr["number"]
    owner     = payload["repository"]["owner"]["login"]
    repo      = payload["repository"]["name"]

    logger.info("Analyzing PR #%d in %s/%s", pr_number, owner, repo)

    try:
        diff   = fetch_pr_diff(owner, repo, pr_number)
        result = DependencyAnalyzer(
            project_id=GCP_PROJECT,
            location=GCP_LOCATION,
        ).analyze_diff(diff)
        comment = format_comment(result)
        post_pr_comment(owner, repo, pr_number, comment)
        return jsonify({"status": "ok", "safe_to_merge": result.safe_to_merge}), 200

    except Exception as exc:
        logger.error("Analysis failed for PR #%d: %s", pr_number, exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))