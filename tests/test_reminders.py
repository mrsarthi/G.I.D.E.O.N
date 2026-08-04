"""Time expression parsing — the logic most likely to silently misfire."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tools.reminders import parse_natural_time

# A Wednesday, deliberately mid-week and mid-afternoon.
NOW = datetime(2026, 8, 5, 14, 30, 0)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("in 5 minutes", NOW + timedelta(minutes=5)),
        ("in 90 minutes", NOW + timedelta(minutes=90)),
        ("in 2 hours", NOW + timedelta(hours=2)),
        ("in 3 days", NOW + timedelta(days=3)),
        ("in 2 weeks", NOW + timedelta(weeks=2)),
    ],
)
def test_relative_offsets(expression, expected):
    assert parse_natural_time(expression, now=NOW) == expected


def test_tomorrow_defaults_to_morning():
    result = parse_natural_time("tomorrow", now=NOW)
    assert result.date() == (NOW + timedelta(days=1)).date()
    assert (result.hour, result.minute) == (9, 0)


def test_tomorrow_honours_explicit_time():
    result = parse_natural_time("tomorrow at 3pm", now=NOW)
    assert result.date() == (NOW + timedelta(days=1)).date()
    assert (result.hour, result.minute) == (15, 0)


def test_tonight_is_evening():
    result = parse_natural_time("tonight", now=NOW)
    assert result.date() == NOW.date()
    assert result.hour == 20


def test_time_already_past_today_rolls_to_tomorrow():
    # 8am requested at 2:30pm means tomorrow morning, not eight hours ago.
    result = parse_natural_time("at 8am", now=NOW)
    assert result > NOW
    assert result.date() == (NOW + timedelta(days=1)).date()
    assert result.hour == 8


def test_time_later_today_stays_today():
    result = parse_natural_time("at 6pm", now=NOW)
    assert result.date() == NOW.date()
    assert result.hour == 18


def test_named_weekday_moves_forward():
    result = parse_natural_time("friday", now=NOW)  # NOW is a Wednesday
    assert result.weekday() == 4
    assert result > NOW


def test_same_weekday_means_next_week():
    result = parse_natural_time("wednesday", now=NOW)
    assert result.weekday() == 2
    assert (result - NOW).days >= 6


def test_midnight_boundary():
    result = parse_natural_time("tomorrow at 12am", now=NOW)
    assert result.hour == 0


def test_noon_boundary():
    result = parse_natural_time("tomorrow at 12pm", now=NOW)
    assert result.hour == 12


def test_unrecognised_expression_is_conservative():
    """An unparseable string must land in the future, never immediately."""
    result = parse_natural_time("whenever you feel like it", now=NOW)
    assert result > NOW
