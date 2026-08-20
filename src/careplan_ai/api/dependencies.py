"""Shared FastAPI dependencies."""

from fastapi import Header

from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import AuthenticationError

API_KEY_HEADER = "X-Api-Key"

# Set by the Java backend to a salted HMAC of the calling staff member's user
# id (the salt is known only to Java), so rate limiting (see core/rate_limit.py)
# can be scoped per person instead of per shared API key - WITHOUT this service
# ever learning or being able to reverse-engineer who that person actually is.
# Treat this purely as an opaque bucketing key; never try to decode it.
CALLER_FINGERPRINT_HEADER = "X-Caller-Fingerprint"


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    settings = get_settings()
    if x_api_key is None or x_api_key != settings.api_key.get_secret_value():
        raise AuthenticationError("Missing or invalid API key")
