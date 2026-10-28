"""Header-based API versioning.

Callers pick the behaviour with `X-CarePlan-Version`; URL paths stay stable.

    1 (or header absent) - the original one-shot generator, POST /v1/care-plans/generate
    2                    - the care plan agent, /care-plans/sessions...

The Java backend sets the header from one config value, so switching or
rolling back is a config change on that side - no frontend or URL change.
"""

from fastapi import Header

from careplan_ai.core.exceptions import UnsupportedVersionError

VERSION_HEADER = "X-CarePlan-Version"
AGENT_VERSION = "2"


async def require_agent_version(
    version: str | None = Header(default=None, alias=VERSION_HEADER),
) -> None:
    if version is None or version.strip() != AGENT_VERSION:
        raise UnsupportedVersionError(
            f"This endpoint needs {VERSION_HEADER}: {AGENT_VERSION} (got {version or 'none'})"
        )
