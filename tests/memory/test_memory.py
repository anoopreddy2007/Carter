"""
Tests for the SQLite memory module (save_note, search_notes, delete_note).
"""

from pathlib import Path

import pytest

from carter.tools.memory import save_note, search_notes, list_notes, delete_note


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "test_carter.db")


class TestSaveNote:
    def test_save_basic_note(self, db: str) -> None:
        result = save_note("This is a test note.", db)
        assert result["id"] > 0
        assert "saved" in result["message"].lower()

    def test_save_note_with_title_and_tags(self, db: str) -> None:
        result = save_note("Content here.", db, title="My Note", tags="test,demo")
        assert result["title"] == "My Note"

    def test_empty_content_raises(self, db: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_note("", db)

    def test_whitespace_only_raises(self, db: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_note("   ", db)

    def test_oversized_note_raises(self, db: str) -> None:
        with pytest.raises(ValueError, match="too long"):
            save_note("x" * 5000, db, max_note_length=100)


class TestSearchNotes:
    def test_search_finds_saved_note(self, db: str) -> None:
        save_note("The quick brown fox", db, title="Fox note")
        result = search_notes("quick", db)
        assert result["count"] >= 1
        assert any("quick" in r["content"] for r in result["results"])

    def test_search_by_title(self, db: str) -> None:
        save_note("Content here.", db, title="UniqueTitle123")
        result = search_notes("UniqueTitle123", db)
        assert result["count"] >= 1

    def test_search_empty_query_raises(self, db: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            search_notes("", db)

    def test_search_no_results(self, db: str) -> None:
        result = search_notes("xyzzy_not_a_real_word", db)
        assert result["count"] == 0
        assert result["results"] == []

    def test_search_limit_respected(self, db: str) -> None:
        for i in range(10):
            save_note(f"Note number {i}", db)
        result = search_notes("Note number", db, limit=3)
        assert len(result["results"]) <= 3


class TestDeleteNote:
    def test_delete_existing_note(self, db: str) -> None:
        saved = save_note("To be deleted.", db)
        note_id = saved["id"]
        result = delete_note(note_id, db)
        assert result["deleted_id"] == note_id

    def test_delete_nonexistent_raises(self, db: str) -> None:
        with pytest.raises(ValueError, match="not found"):
            delete_note(99999, db)

    def test_deleted_note_not_found_in_search(self, db: str) -> None:
        saved = save_note("Unique_delete_test_content_abc123", db)
        note_id = saved["id"]
        delete_note(note_id, db)
        result = search_notes("Unique_delete_test_content_abc123", db)
        assert result["count"] == 0


class TestPersistence:
    def test_notes_persist_across_connections(self, db: str) -> None:
        """Notes saved in one connection should be visible in another."""
        save_note("Persistent note content.", db, title="Persistent")
        # Open a new connection to the same file
        result = search_notes("Persistent note", db)
        assert result["count"] >= 1
