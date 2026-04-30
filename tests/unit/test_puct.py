"""Tests for the PUCT scoring formula."""

from __future__ import annotations

from muzero.search.puct import puct_score


def test_puct_score_increases_with_prior() -> None:
    """Higher prior should produce higher PUCT score, all else equal."""
    score_low = puct_score(
        parent_visit_count=10,
        child_visit_count=2,
        child_prior=0.1,
        child_value=0.5,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    score_high = puct_score(
        parent_visit_count=10,
        child_visit_count=2,
        child_prior=0.9,
        child_value=0.5,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    assert score_high > score_low


def test_puct_score_decreases_with_child_visits() -> None:
    """More visits to a child should decrease its PUCT score (exploration term)."""
    score_few = puct_score(
        parent_visit_count=10,
        child_visit_count=1,
        child_prior=0.5,
        child_value=0.5,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    score_many = puct_score(
        parent_visit_count=10,
        child_visit_count=5,
        child_prior=0.5,
        child_value=0.5,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    # With more visits, the exploration bonus should be smaller
    assert score_few > score_many


def test_puct_score_includes_child_value() -> None:
    """Higher value should produce higher PUCT score."""
    score_low = puct_score(
        parent_visit_count=10,
        child_visit_count=2,
        child_prior=0.5,
        child_value=-1.0,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    score_high = puct_score(
        parent_visit_count=10,
        child_visit_count=2,
        child_prior=0.5,
        child_value=1.0,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    assert score_high > score_low


def test_puct_score_zero_visits() -> None:
    """PUCT should work with zero child visits (initial state)."""
    score = puct_score(
        parent_visit_count=1,
        child_visit_count=0,
        child_prior=0.5,
        child_value=0.0,
        pb_c_base=19652,
        pb_c_init=1.25,
    )
    assert isinstance(score, float)
    assert score > 0  # Should have positive exploration bonus
