"""
main.py
Flask webhook server — receives GitHub PR events,
triggers dependency analysis, posts result as PR comment.

Run locally:
    flask run --port 8080

Run in production (Cloud Run):
    gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 120 main:app
"""

import hashlib
import hmac
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Optional

from flask import Flask, Response, jsonify, request

from analyzer import DependencyAnalyzer, AnalysisResult

# ---------------------------------------------------------------------------
# Logging — stdout so Cloud Run captures it
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

# GCP / Vertex AI
GCP_PROJECT          = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION         = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_API_ENDPOINT  = os.environ.get("VERTEX_API_ENDPOINT", "")      # custom endpoint if needed
VERTEX_MODEL         = os.environ.get("VERTEX_MODEL", "gemini-1.5-pro")

# Token broker (your local token API)
TOKEN_BROKER_URL     = os.environ.get("TOKEN_BROKER_URL", "")
TOKEN_BROKER_PROXY   = os.environ.get("TOKEN_BROKER_PROXY", "")       # http://proxy.corp.com:8080
TOKEN_BROKER_API_KEY = os.environ.get("TOKEN_BROKER_API_KEY", "")

# GitHub
GITHUB_TOKEN         = os.environ.get("GITHUB_TOKEN", "")
GITHUB_PROXY         = os.environ.get("GITHUB_PROXY", "")             # http://proxy.corp.com:8080
GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")  # for signature verification

# Server
PORT                 = int(os.environ.get("PORT", "8080"))

# ---------------------------------------------------------------------------
# Startup validation — fail fast if required config is missing
# ---------------------------------------------------------------------------

_REQUIRED = {
    "GCP_PROJECT":      GCP_PROJECT,
    "GITHUB_TOKEN":     GITHUB_TOKEN,
    "TOKEN_BROKER_URL": TOKEN_BROKER_URL,
}

def _validate_config() -> None:
    missing = [k for k, v in _REQUIRED.items() if not v]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Token fetcher — calls your local/internal token broker API
# ---------------------------------------------------------------------------

def get_token() -> str:
    logger.info("Fetching token from broker: %s", TOKEN_BROKER_URL)

    if TOKEN_BROKER_PROXY:
        proxy_handler = urllib.request.ProxyHandler({
            "http":  TOKEN_BROKER_PROXY,
            "https": TOKEN_BROKER_PROXY,
        })
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    headers = {}
    if TOKEN_BROKER_API_KEY:
        headers["X-Api-Key"] = TOKEN_BROKER_API_KEY

    req = urllib.request.Request(TOKEN_BROKER_URL, headers=headers)

    try:
        with opener.open(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            logger.info("Token fetched successfully.")
            return data["token"]   # adjust key to match your broker's response
    except Exception as e:
        logger.error("Failed to fetch token: %s", e)
        raise

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _github_opener() -> urllib.request.OpenerDirector:
    if GITHUB_PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({
                "http":  GITHUB_PROXY,
                "https": GITHUB_PROXY,
            })
        )
    return urllib.request.build_opener()


def _github_request(url: str, method: str = "GET", body: Optional[dict] = None) -> dict:
    headers = {
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(body).encode() if body else None
    if data:
        headers["Content-Type"] = "application/json"

    req    = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = _github_opener()

    try:
        with opener.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        logger.error("GitHub API error %s %s: %s", e.code, url, e.read().decode())
        raise
    except urllib.error.URLError as e:
        logger.error("GitHub connection error %s: %s", url, e.reason)
        raise


def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    opener = _github_opener()
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.error("Failed to fetch PR diff #%d: %s", pr_number, e.read().decode())
        raise


def post_pr_comment(owner: str, repo: str, pr_number: int, body: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    _github_request(url, method="POST", body={"body": body})
    logger.info("Posted comment to PR #%d", pr_number)

# ---------------------------------------------------------------------------
# Webhook signature verification
# ---------------------------------------------------------------------------

def _verify_signature(payload: bytes, signature: str) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        logger.warning("No webhook secret configured — skipping verification.")
        return True
    expected = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

# ---------------------------------------------------------------------------
# Comment formatter
# ---------------------------------------------------------------------------

def format_comment(result: AnalysisResult) -> str:
    icon    = "✅" if result.safe_to_merge else "🚨"
    verdict = "SAFE TO MERGE" if result.safe_to_merge else "REVIEW REQUIRED"
    lines   = [f"{icon} **Dependency Analysis: {verdict}**\n"]

    if result.dependency_changes:
        lines.append("| Package | Old | New | File |")
        lines.append("|---|---|---|---|")
        for d in result.dependency_changes:
            lines.append(f"| `{d.name}` | {d.old_version} | {d.new_version} | `{d.file}` |")
        lines.append("")

    sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}

    if result.breaking_changes:
        lines.append("### Breaking Changes")
        for bc in result.breaking_changes:
            icon_sev = sev_icon.get(bc.severity.value, "⚪")
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
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "healthy"}), 200


@app.route("/webhook", methods=["POST"])
def webhook() -> Response:
    # 1. Verify GitHub signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_signature(request.data, signature):
        logger.warning("Invalid webhook signature.")
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Only handle pull_request events
    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return jsonify({"status": "ignored", "event": event}), 200

    # 3. Only act on relevant actions
    payload = request.json
    action  = payload.get("action", "")
    if action not in ("opened", "synchronize", "reopened"):
        return jsonify({"status": "ignored", "action": action}), 200

    # 4. Extract PR details
    pr        = payload["pull_request"]
    pr_number = pr["number"]
    owner     = payload["repository"]["owner"]["login"]
    repo      = payload["repository"]["name"]
    logger.info("Handling PR #%d in %s/%s", pr_number, owner, repo)

    # 5. Fetch diff
    try:
        diff = fetch_pr_diff(owner, repo, pr_number)
    except Exception as e:
        logger.error("Failed to fetch diff: %s", e)
        return jsonify({"error": "Failed to fetch PR diff"}), 500

    # 6. Analyze
    try:
        analyzer = DependencyAnalyzer(
            project_id=GCP_PROJECT,
            location=GCP_LOCATION,
            model_name=VERTEX_MODEL,
            api_endpoint=VERTEX_API_ENDPOINT,
            token_fetcher=get_token,         # injected — no hardcoding
        )
        result = analyzer.analyze_diff(diff)
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        return jsonify({"error": "Analysis failed"}), 500

    # 7. Post comment
    try:
        post_pr_comment(owner, repo, pr_number, format_comment(result))
    except Exception as e:
        logger.error("Failed to post comment: %s", e)
        # Don't fail the webhook — analysis succeeded

    return jsonify({
        "status":        "ok",
        "pr":            pr_number,
        "safe_to_merge": result.safe_to_merge,
    }), 200

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _validate_config()
    logger.info("Starting server on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)