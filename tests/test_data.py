"""Tests for data.py — dataset integrity and structure."""

from ad_review_env.data import CONTENT_ITEMS, CONTENT_INDEX
from ad_review_env.models import IAB_CATEGORIES, GARM_CATEGORIES


def test_dataset_has_50_items():
    assert len(CONTENT_ITEMS) == 50


def test_difficulty_distribution():
    by_diff = {}
    for item in CONTENT_ITEMS:
        by_diff.setdefault(item["difficulty"], []).append(item)
    assert len(by_diff["easy"]) == 15
    assert len(by_diff["medium"]) == 18
    assert len(by_diff["hard"]) == 17


def test_content_ids_are_unique():
    ids = [item["content_id"] for item in CONTENT_ITEMS]
    assert len(ids) == len(set(ids))


def test_content_index_matches():
    assert len(CONTENT_INDEX) == 50
    for item in CONTENT_ITEMS:
        assert item["content_id"] in CONTENT_INDEX
        assert CONTENT_INDEX[item["content_id"]] is item


def test_required_fields_present():
    required = [
        "content_id", "content_text", "content_type", "platform",
        "difficulty", "gold_decision", "gold_iab_category", "gold_garm_category",
        "gold_risk_level", "gold_age_rating",
    ]
    for item in CONTENT_ITEMS:
        for field in required:
            assert field in item, f"Missing {field} in {item['content_id']}"


def test_gold_decisions_valid():
    valid = {"APPROVE", "REJECT", "ESCALATE"}
    for item in CONTENT_ITEMS:
        assert item["gold_decision"] in valid, (
            f"{item['content_id']}: invalid gold_decision '{item['gold_decision']}'"
        )


def test_gold_iab_categories_valid():
    for item in CONTENT_ITEMS:
        assert item["gold_iab_category"] in IAB_CATEGORIES, (
            f"{item['content_id']}: invalid gold_iab_category '{item['gold_iab_category']}'"
        )


def test_gold_garm_categories_valid():
    for item in CONTENT_ITEMS:
        assert item["gold_garm_category"] in GARM_CATEGORIES, (
            f"{item['content_id']}: invalid gold_garm_category '{item['gold_garm_category']}'"
        )


def test_gold_risk_levels_valid():
    valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for item in CONTENT_ITEMS:
        assert item["gold_risk_level"] in valid, (
            f"{item['content_id']}: invalid gold_risk_level '{item['gold_risk_level']}'"
        )


def test_platforms_valid():
    valid = {"instagram", "x", "youtube", "tiktok", "facebook", "reddit", "threads", "linkedin"}
    for item in CONTENT_ITEMS:
        assert item["platform"] in valid, (
            f"{item['content_id']}: unexpected platform '{item['platform']}'"
        )


def test_content_types_valid():
    valid = {"post", "comment", "caption", "bio", "reel", "story", "thread"}
    for item in CONTENT_ITEMS:
        assert item["content_type"] in valid, (
            f"{item['content_id']}: unexpected content_type '{item['content_type']}'"
        )


def test_content_text_not_empty():
    for item in CONTENT_ITEMS:
        assert len(item["content_text"].strip()) > 0, (
            f"{item['content_id']}: empty content_text"
        )


def test_age_ratings_valid():
    valid_age_ratings = {"ALL_AGES", "TEEN", "MATURE", "ADULT"}
    for item in CONTENT_ITEMS:
        assert item["gold_age_rating"] in valid_age_ratings, (
            f"{item['content_id']}: invalid gold_age_rating '{item['gold_age_rating']}'"
        )
