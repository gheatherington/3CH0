"""Standard per-user locations (via platformdirs). Override with ECHO360_DL_HOME.

* data_dir/browser-profile   persistent Playwright profile (login cookies live here)
* cache_dir/models           speech-recognition models downloaded on request
* cache_dir/discovery        sanitised discovery traces (``inspect --save-trace``)
* default output             ~/Downloads (or the platform's download dir)
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "echo360-dl"


def _dirs() -> PlatformDirs:
    return PlatformDirs(APP_NAME, appauthor=False, ensure_exists=False)


def _override() -> Path | None:
    value = os.environ.get("ECHO360_DL_HOME")
    return Path(value).expanduser() if value else None


def data_dir() -> Path:
    base = _override()
    return base / "data" if base else Path(_dirs().user_data_dir)


def cache_dir() -> Path:
    base = _override()
    return base / "cache" if base else Path(_dirs().user_cache_dir)


def config_dir() -> Path:
    base = _override()
    return base / "config" if base else Path(_dirs().user_config_dir)


def browser_profile_dir() -> Path:
    return data_dir() / "browser-profile"


def models_dir() -> Path:
    return cache_dir() / "models"


def default_output_dir() -> Path:
    downloads = Path(_dirs().user_downloads_dir)
    return downloads if downloads.is_dir() else Path.cwd()


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) with 0700 permissions on the leaf."""
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    return path


#: Written into every browser profile echo360-dl creates. Only such directories (or the
#: default location) are used or deleted, so a real Chrome profile is never touched.
PROFILE_MARKER = ".echo360-dl-profile"


def is_own_profile(path: Path) -> bool:
    path = Path(path)
    return (path / PROFILE_MARKER).is_file() or (
        path.resolve() == browser_profile_dir().resolve())


def prepare_profile_dir(path: Path) -> Path:
    """Create/validate a dedicated browser profile directory (0700, marked as ours).

    Raises ValueError for an existing non-empty directory that echo360-dl did not create
    (for example a real Chrome profile passed via ``--profile-dir``).
    """
    path = Path(path).expanduser()
    if path.exists() and (not path.is_dir() or (any(path.iterdir())
                                                and not is_own_profile(path))):
        raise ValueError(f"{path} is not an echo360-dl browser profile")
    ensure_private_dir(path)
    marker = path / PROFILE_MARKER
    if not marker.exists():
        marker.write_text("Dedicated echo360-dl browser profile. Safe to delete with "
                          "`echo360-dl auth logout`.\n", encoding="utf-8")
    return path
