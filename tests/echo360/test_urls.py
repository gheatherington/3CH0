from __future__ import annotations

import pytest

from echo360_dl.echo360.urls import (
    SUPPORTED_HOSTS,
    content_path_ids,
    is_echo360_url,
    is_login_url,
    lesson_param,
    media_param,
    parse_classroom_url,
)
from echo360_dl.errors import InvalidURLError

from .conftest import CLASSROOM, INSTITUTION, LESSON, MEDIA, playlist_url


def test_parse_good_url():
    ref = parse_classroom_url(CLASSROOM)
    assert ref.host == "echo360.ca"
    assert ref.lesson_id == LESSON
    assert ref.url == CLASSROOM


@pytest.mark.parametrize("suffix", ["?timestamp=1234", "?timestamp=00:10:00#frag", "/",
                                    "/?foo=bar&timestamp=99", "#t=10"])
def test_query_fragment_and_timestamp_are_dropped(suffix):
    ref = parse_classroom_url(CLASSROOM + suffix)
    assert ref.url == CLASSROOM
    assert "timestamp" not in ref.url and "?" not in ref.url and "#" not in ref.url
    # No trimming information is carried anywhere on the reference.
    assert set(vars(ref)) == {"host", "lesson_id", "url"}


def test_whitespace_and_case_insensitive_host():
    assert parse_classroom_url(f"  https://ECHO360.CA/lesson/{LESSON}/classroom \n").lesson_id \
        == LESSON


@pytest.mark.parametrize("url", [
    "",
    "not a url",
    f"http://echo360.ca/lesson/{LESSON}/classroom",
    f"https://echo360.org/lesson/{LESSON}/classroom",
    f"https://echo360.org.uk/lesson/{LESSON}/classroom",
    f"https://echo360.com/lesson/{LESSON}/classroom",
    f"https://evil.example/lesson/{LESSON}/classroom",
    f"https://echo360.ca.evil.example/lesson/{LESSON}/classroom",
    f"https://content.echo360.ca/lesson/{LESSON}/classroom",
    f"https://user:pw@echo360.ca/lesson/{LESSON}/classroom",
    f"https://echo360.ca:8443/lesson/{LESSON}/classroom",
    "https://echo360.ca/lesson//classroom",
    f"https://echo360.ca/lesson/{LESSON}",
    f"https://echo360.ca/lesson/{LESSON}/classroom/extra",
    "https://echo360.ca/section/abc/home",
    "https://echo360.ca/lesson/%2E%2E%2Fetc/classroom",
    "ftp://echo360.ca/lesson/x/classroom",
])
def test_bad_urls_rejected_with_hint(url):
    with pytest.raises(InvalidURLError) as info:
        parse_classroom_url(url)
    assert info.value.hint and "classroom" in info.value.hint


def test_only_echo360_ca_is_claimed():
    assert frozenset({"echo360.ca"}) == SUPPORTED_HOSTS


@pytest.mark.parametrize(("url", "expected"), [
    ("https://echo360.ca/login", True),
    ("https://echo360.ca/directLogin?x=1", True),
    ("https://echo360.ca/sso/saml/start", True),
    ("https://echo360.ca/auth/callback", True),
    ("https://login.microsoftonline.com/common/oauth2/authorize", True),
    ("https://idp.university.example/idp/profile/SAML2/Redirect/SSO", True),
    (CLASSROOM, False),
    ("https://echo360.ca/courses", False),
    ("https://echo360.ca/section/abc/home", False),
    ("about:blank", False),
])
def test_login_detection(url, expected):
    assert is_login_url(url) is expected


def test_content_url_helpers():
    url = playlist_url(2)
    assert is_echo360_url(url)
    assert not is_echo360_url("http://content.echo360.ca/x")
    assert lesson_param(url) == LESSON and media_param(url) == MEDIA
    assert content_path_ids(url) == (INSTITUTION, MEDIA)
    assert content_path_ids("https://content.echo360.ca/a/b/s1_av.m3u8") == (None, None)


@pytest.mark.parametrize(("url", "expected"), [
    ("https://login.microsoftonline.com/common/oauth2/authorize", True),
    ("https://sso.university.example/start", True),
    ("https://idp.university.example/x", True),
    ("https://echo360.ca/login", True),
    ("https://cdn.foreign.example/x", False),
    ("https://echo360.ca/api/x", False),
])
def test_looks_like_login_page(url, expected):
    from echo360_dl.echo360.urls import looks_like_login_page

    assert looks_like_login_page(url) is expected


# Synthetic id with the observed echo360.ca shape: G_<uuid>_<uuid>_<start>_<end>.
REAL_SHAPE_ID = ("G_11111111-2222-3333-4444-555555555555_66666666-7777-8888-9999-000000000000_"
                 "2026-01-05T10:30:00.000_2026-01-05T11:20:00.000")


@pytest.mark.parametrize("encoded", [False, True])
def test_observed_lesson_id_shape_accepted(encoded):
    from urllib.parse import quote

    path_id = quote(REAL_SHAPE_ID, safe="") if encoded else REAL_SHAPE_ID
    ref = parse_classroom_url(f"https://echo360.ca/lesson/{path_id}/classroom?timestamp=12")
    assert ref.lesson_id == REAL_SHAPE_ID and len(REAL_SHAPE_ID) == 123


def test_short_id_distinguishes_lessons_sharing_a_prefix():
    from echo360_dl.models import Recording

    other = REAL_SHAPE_ID.replace("10:30:00", "14:30:00")
    a = Recording(lesson_id=REAL_SHAPE_ID, classroom_url="u", host="echo360.ca")
    b = Recording(lesson_id=other, classroom_url="u", host="echo360.ca")
    assert a.short_id != b.short_id and len(a.short_id) == 10
    assert a.short_id == Recording(lesson_id=REAL_SHAPE_ID, classroom_url="x",
                                   host="echo360.ca").short_id
