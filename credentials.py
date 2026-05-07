import urllib.request
import json
import logging
import os
from datetime import datetime, timedelta
from google.auth.credentials import Credentials as BaseCredentials
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)

# Set this as an env var in Cloud Run — never hardcode
TOKEN_BROKER_URL = os.environ.get("TOKEN_BROKER_URL", "http://your-internal-api/token")


def get_token() -> str:
    """
    Calls your internal token broker API to fetch a fresh token.
    TOKEN_BROKER_URL is injected via environment variable.
    """
    logger.info("Fetching token from broker: %s", TOKEN_BROKER_URL)
    try:
        req = urllib.request.Request(
            TOKEN_BROKER_URL,
            headers={
                # Add any auth your broker needs — e.g. an API key
                "X-Api-Key": os.environ.get("TOKEN_BROKER_API_KEY", ""),
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["token"]   # adjust key to match your API response shape
    except Exception as e:
        logger.error("Failed to fetch token from broker: %s", e)
        raise


class RefreshableCredentials(BaseCredentials):
    def __init__(self, token_fetcher: callable, expiry_seconds: int = 3600):
        super().__init__()
        self._token_fetcher   = token_fetcher
        self._expiry_seconds  = expiry_seconds
        self._refresh_token()

    def _refresh_token(self) -> None:
        logger.info("Refreshing token...")
        self.token  = self._token_fetcher()
        self.expiry = datetime.utcnow() + timedelta(seconds=self._expiry_seconds - 60)
        logger.info("Token refreshed. Expires at: %s", self.expiry)

    def refresh(self, request: Request) -> None:
        self._refresh_token()

    @property
    def valid(self) -> bool:
        return self.token is not None and not self.expired