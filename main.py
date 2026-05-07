import urllib.request
import urllib.error
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── env vars ──────────────────────────────────────────────────────────────
GITHUB_TOKEN  = os.environ["GITHUB_TOKEN"]
GITHUB_PROXY  = os.environ.get("GITHUB_PROXY", "")   # e.g. http://proxy.corp.com:8080


def _github_opener() -> urllib.request.OpenerDirector:
    """
    Builds a urllib opener with proxy support if GITHUB_PROXY is set.
    Reuse this for all GitHub API calls.
    """
    if GITHUB_PROXY:
        proxy_handler = urllib.request.ProxyHandler({
            "http":  GITHUB_PROXY,
            "https": GITHUB_PROXY,
        })
        return urllib.request.build_opener(proxy_handler)
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
        logger.error("GitHub API error %s: %s — %s", e.code, url, e.read().decode())
        raise
    except urllib.error.URLError as e:
        logger.error("GitHub connection error: %s — %s", url, e.reason)
        raise


def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={
        "Authorization":        f"Bearer {GITHUB_TOKEN}",
        "Accept":               "application/vnd.github.v3.diff",   # raw diff
        "X-GitHub-Api-Version": "2022-11-28",
    })
    opener = _github_opener()
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.error("Failed to fetch PR diff %s: %s", pr_number, e.read().decode())
        raise


def post_pr_comment(owner: str, repo: str, pr_number: int, body: str) -> None:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    _github_request(url, method="POST", body={"body": body})
    logger.info("Posted comment to PR #%d", pr_number)