"""Open Care Buddy prompt files (OPENCARE-102).

The prompt wording lives in files, not in code, so it can be changed without a release:

    chat_prompt.md      section 1 of OpenCare_Buddy_AI_Prompts_24-11-2026.md, with ${...}
                        placeholders filled from each /buddy/chat request
    open_care_guide.md  how the app works (tabs, charts, click paths), kept with each
                        Open Care release; filled into ${open_care_guide}

They ship in the package's prompts/ folder; CAREPLAN_AI_BUDDY_PROMPT_DIR points the service
at another folder holding the same file names.
"""

import re
from functools import lru_cache
from pathlib import Path

from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError

CHAT_PROMPT = "chat_prompt.md"
OPEN_CARE_GUIDE = "open_care_guide.md"

PLACEHOLDER = re.compile(r"\$\{([a-z_]+)\}")
_PACKAGED = Path(__file__).parent / "prompts"


def prompt_dir() -> Path:
    configured = get_settings().buddy_prompt_dir
    return Path(configured) if configured else _PACKAGED


@lru_cache
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").rstrip("\n")
    except OSError as exc:
        raise ConfigurationError(f"Open Care Buddy prompt file not found: {path}") from exc


def load(name: str) -> str:
    return _read(prompt_dir() / name)


def clear_cache() -> None:
    _read.cache_clear()
