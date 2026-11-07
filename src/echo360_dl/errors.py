"""Exception hierarchy and documented process exit codes.

Every error raised deliberately by echo360-dl derives from :class:`Echo360DLError`
and carries an exit code, a stable machine-readable ``kind`` and an optional
actionable ``hint``. Messages must never contain credentials, cookies or raw
request headers (see :mod:`echo360_dl.redact`).
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes. Documented in README.md -> "Exit codes"."""

    OK = 0
    ERROR = 1  # unexpected/internal error
    USAGE = 2  # invalid arguments, or a required choice missing in non-interactive mode
    AUTH = 3  # login required, session expired, or authorization refused
    UNAVAILABLE = 4  # requested content (source, quality, language, transcript) not available
    PARTIAL = 5  # some requested outputs were produced, others failed
    DEPENDENCY = 6  # missing external tool, browser, optional package or model
    NETWORK = 7  # transient network/server failure persisted after bounded retries
    VERIFY = 8  # output produced but failed ffprobe verification / validation
    CANCELLED = 130  # interrupted by the user (SIGINT)


class Echo360DLError(Exception):
    """Base class. ``kind`` is a stable slug used in JSON output."""

    exit_code: ExitCode = ExitCode.ERROR
    kind: str = "error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "message": self.message, "hint": self.hint,
                "exit_code": int(self.exit_code)}


class UsageError(Echo360DLError):
    exit_code = ExitCode.USAGE
    kind = "usage"


class InvalidURLError(UsageError):
    kind = "invalid-url"


class SelectionRequiredError(UsageError):
    """A choice is required but prompting is not allowed (non-interactive)."""

    kind = "selection-required"


class AuthError(Echo360DLError):
    exit_code = ExitCode.AUTH
    kind = "auth"


class LoginRequiredError(AuthError):
    """No usable session; the user must run ``3ch0 auth login``."""

    kind = "login-required"


class AuthExpiredError(AuthError):
    """Credentials existed but have expired and bounded refresh failed."""

    kind = "auth-expired"


class UnauthorizedError(AuthError):
    """The server refused access with a definitive authorization response."""

    kind = "unauthorized"


class ForbiddenError(AuthError):
    """HTTP 403 that persisted (after a refresh where possible). Echo360/CloudFront answer
    403 both for inaccessible and for absent objects, so this never claims absence."""

    kind = "forbidden"


class UnavailableError(Echo360DLError):
    exit_code = ExitCode.UNAVAILABLE
    kind = "unavailable"


class QualityUnavailableError(UnavailableError):
    kind = "quality-unavailable"


class LanguageUnavailableError(UnavailableError):
    kind = "language-unavailable"


class TranscriptUnavailableError(UnavailableError):
    kind = "transcript-unavailable"


class UnsupportedError(UnavailableError):
    """Content exists but uses a format/protection this tool does not handle (e.g. DRM)."""

    kind = "unsupported"


class DependencyError(Echo360DLError):
    exit_code = ExitCode.DEPENDENCY
    kind = "dependency"


class ModelDownloadRequiredError(DependencyError):
    """A speech model is not cached and downloading it was not explicitly allowed."""

    kind = "model-download-required"


class NetworkError(Echo360DLError):
    exit_code = ExitCode.NETWORK
    kind = "network"


class VerificationError(Echo360DLError):
    exit_code = ExitCode.VERIFY
    kind = "verification"


class ValidationError(VerificationError):
    """Invalid user-provided data (chapter file, subtitle file)."""

    kind = "validation"


class PartialFailureError(Echo360DLError):
    exit_code = ExitCode.PARTIAL
    kind = "partial"


class CancelledError(Echo360DLError):
    exit_code = ExitCode.CANCELLED
    kind = "cancelled"
