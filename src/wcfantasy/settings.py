"""Local UI settings for the web app (currently just the language).

The advisor engine is language-agnostic; only the web GUI text is
translated, in the frontend. This persists the chosen language so it
survives a restart. Code, comments and identifiers stay English (repo
convention): only the visible UI strings switch.
"""

import json
from dataclasses import asdict, dataclass, fields

from .config import STATE_DIR

UI_SETTINGS_FILE = STATE_DIR / "ui_settings.json"
SUPPORTED_LANGUAGES = ("en", "it")
DEFAULT_LANGUAGE = "en"


@dataclass
class UiSettings:
    language: str = DEFAULT_LANGUAGE


def load() -> UiSettings:
    """Load persisted UI settings, tolerant of a missing/old/corrupt file."""
    if not UI_SETTINGS_FILE.exists():
        return UiSettings()
    try:
        data = json.loads(UI_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return UiSettings()
    known = {f.name for f in fields(UiSettings)}
    settings = UiSettings(**{k: v for k, v in data.items() if k in known})
    if settings.language not in SUPPORTED_LANGUAGES:
        settings.language = DEFAULT_LANGUAGE
    return settings


def save(settings: UiSettings) -> None:
    UI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    UI_SETTINGS_FILE.write_text(
        json.dumps(asdict(settings), indent=2), encoding="utf-8"
    )


def set_language(language: str) -> UiSettings:
    """Persist the chosen language; raises ValueError on an unsupported one."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}; choose from "
            f"{', '.join(SUPPORTED_LANGUAGES)}."
        )
    settings = load()
    settings.language = language
    save(settings)
    return settings
