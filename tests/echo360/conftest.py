"""Shared fictional values and helpers for echo360 tests. Nothing here is a real secret."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "echo360"

# Clearly fictional identifiers.
LESSON = "G_11111111-2222-4333-8444-555555555555_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
OTHER_LESSON = "G_99999999-2222-4333-8444-555555555555_aaaaaaaa-bbbb-4ccc-8ddd-000000000000"
MEDIA = "99999999-8888-4777-8666-555555555555"
OTHER_MEDIA = "12121212-3434-4565-8787-909090909090"
INSTITUTION = "00000000-1111-4222-8333-444444444444"
CONTENT_BASE = f"https://content.echo360.ca/0000.{INSTITUTION}/{MEDIA}/1"
SECRET_QUERY = (f"x-uid=FAKEUSER000&x-sid=FAKESESSION000&x-lid={LESSON}&x-mid={MEDIA}"
                "&x-instid=FAKEINST000&x-oid=FAKEORG000&x-did=FAKEDEVICE000&x-act=play"
                "&x-src=fake")
CLASSROOM = f"https://echo360.ca/lesson/{LESSON}/classroom"

FAKE_SIGNATURE = "FAKE-SIGNATURE-VALUE-0123456789"
FAKE_JWT = "FAKE-JWT-VALUE-ABCDEFGHIJ"


def playlist_url(index: int, *, lesson: str = LESSON, media: str = MEDIA) -> str:
    query = SECRET_QUERY.replace(LESSON, lesson).replace(MEDIA, media)
    return (f"https://content.echo360.ca/0000.{INSTITUTION}/{media}/1/s{index}_av.m3u8?"
            f"{query}")


def cloudfront_policy(epoch: int, media: str = MEDIA) -> str:
    policy = {"Statement": [{"Resource": f"https://*.echo360.ca/*{media}/*",
                             "Condition": {"DateLessThan": {"AWS:EpochTime": epoch}}}]}
    raw = base64.b64encode(json.dumps(policy).encode()).decode()
    return raw.replace("+", "-").replace("=", "_").replace("/", "~")


def master_text(index: int) -> str:
    return (FIXTURES / f"master_s{index}.m3u8").read_text()


class RecordingProgress:
    """ProgressSink that records messages for assertions."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def stage(self, stage, message=None):
        self.messages.append(("stage", str(message)))

    def task_start(self, task_id, description, total=None, unit="bytes"):
        self.messages.append(("task_start", description))

    def task_update(self, task_id, **kwargs):
        pass

    def task_done(self, task_id, *, ok=True, message=None):
        pass

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))

    def debug(self, message):
        self.messages.append(("debug", message))


@pytest.fixture
def progress() -> RecordingProgress:
    return RecordingProgress()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHO360_DL_HOME", str(tmp_path / "home"))
