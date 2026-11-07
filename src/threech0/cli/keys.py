"""Keyboard input for the live console (POSIX terminals only).

The terminal is put in *cbreak* mode (no echo, no line buffering) while the live display
runs. Signal keys stay enabled, so Ctrl-C still sends SIGINT and keeps its existing
meaning (first press: stop after the current step; second: abort). The saved terminal
attributes are restored by :meth:`KeyReader.stop`, which the live display calls on
success, failure, interruption and before every prompt; an ``atexit`` hook is the last
resort.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import select
import threading
from collections.abc import Callable
from typing import IO, Any

try:  # pragma: no cover - always present on macOS/Linux
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

#: Key names passed to the callback.
UP, DOWN, ENTER, DETAILS = "up", "down", "enter", "details"

_SEQUENCES = {b"\x1b[A": UP, b"\x1bOA": UP, b"\x1b[B": DOWN, b"\x1bOB": DOWN,
              b"k": UP, b"j": DOWN, b"\r": ENTER, b"\n": ENTER, b"d": DETAILS,
              b"D": DETAILS}


def parse_keys(data: bytes) -> list[str]:
    """Split raw terminal input into key names (unknown bytes are ignored)."""
    keys: list[str] = []
    i = 0
    while i < len(data):
        for length in (3, 1):
            chunk = data[i:i + length]
            if chunk in _SEQUENCES:
                keys.append(_SEQUENCES[chunk])
                i += length
                break
        else:
            i += 3 if data[i:i + 1] == b"\x1b" and data[i + 1:i + 2] in (b"[", b"O") else 1
    return keys


class KeyReader:
    def __init__(self, stream: IO[str], on_key: Callable[[str], None]) -> None:
        self._stream = stream
        self._on_key = on_key
        self._fd: int | None = None
        self._saved: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self) -> bool:
        """Enter cbreak mode and read keys in the background; False if unsupported."""
        if termios is None or tty is None:
            return False
        with self._lock:
            if self._thread is not None:
                return True
            try:
                fd = self._stream.fileno()
                if not os.isatty(fd):
                    return False
                saved = termios.tcgetattr(fd)
                tty.setcbreak(fd, termios.TCSANOW)
            except (AttributeError, OSError, ValueError, termios.error):
                return False
            self._fd, self._saved = fd, saved
            _ACTIVE.add(self)
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="3ch0-keys", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            thread, self._thread = self._thread, None
            self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._restore()

    def _restore(self) -> None:
        fd, saved = self._fd, self._saved
        self._fd = self._saved = None
        _ACTIVE.discard(self)
        if fd is not None and saved is not None and termios is not None:
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _loop(self) -> None:
        fd = self._fd
        while fd is not None and not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(fd, 64)
            except (OSError, ValueError):
                return
            if not data:
                return
            for key in parse_keys(data):
                with contextlib.suppress(Exception):
                    self._on_key(key)


_ACTIVE: set[KeyReader] = set()


@atexit.register
def _restore_all() -> None:  # pragma: no cover - safety net
    for reader in list(_ACTIVE):
        reader._restore()
