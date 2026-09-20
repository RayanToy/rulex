"""Модульные тесты чистых функций: без сети и без обращений к LLM."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import metrics  # noqa: E402

from auth import hash_password, needs_rehash, verify_password  # noqa: E402
from main import calculate_level  # noqa: E402


class TestPasswordHashing:
    def test_roundtrip(self):
        stored = hash_password("правильный-пароль")
        assert stored.startswith("$argon2")
        assert verify_password("правильный-пароль", stored)
        assert not verify_password("другой-пароль", stored)

    def test_salt_is_random(self):
        assert hash_password("одинаковый") != hash_password("одинаковый")

    def test_legacy_sha256_still_verifies(self):
        """Пользователи, заведённые до перехода на Argon2, должны входить."""
        import hashlib
        salt = "a" * 32
        legacy = f"{salt}${hashlib.sha256(('старый-пароль' + salt).encode()).hexdigest()}"
        assert verify_password("старый-пароль", legacy)
        assert not verify_password("неверный", legacy)

    def test_legacy_hash_is_marked_for_rehash(self):
        import hashlib
        salt = "b" * 32
        legacy = f"{salt}${hashlib.sha256(('пароль' + salt).encode()).hexdigest()}"
        assert needs_rehash(legacy) is True
        assert needs_rehash(hash_password("пароль")) is False

    @pytest.mark.parametrize("broken", ["", "мусор", "нет-разделителя", "$argon2id$сломано"])
    def test_broken_hash_rejects_instead_of_raising(self, broken):
        assert verify_password("что-нибудь", broken) is False


class TestLevel:
    @pytest.mark.parametrize("pct,high,low,expected", [
        (95.0, 100.0, 80.0, "high"),
        (95.0, 100.0, 50.0, "medium"),   # мало редких слов -> не высокий
        (75.0, 90.0, 40.0, "medium"),
        (50.0, 60.0, 10.0, "low"),
    ])
    def test_thresholds(self, pct, high, low, expected):
        assert calculate_level(pct, high, low) == expected


class TestMetrics:
    def test_confusion_counts(self):
        result = metrics.confusion(
            ["artifact", "artifact", "real", "real"],
            ["artifact", "real", "real", "artifact"])
        assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
        assert result["precision"] == result["recall"] == result["f1"] == 0.5

    def test_confusion_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            metrics.confusion(["real"], ["real", "real"])

    def test_leaked_words_detects_same_lemma(self):
        assert metrics.leaked_words("Главный город страны", ["города"]) == ["город"]

    def test_leaked_words_ignores_unrelated(self):
        assert metrics.leaked_words("Крупное поселение", ["город"]) == []

    def test_pos_match_ratio(self):
        assert metrics.pos_match_ratio("бежать", ["идти", "ходить", "стол"]) == pytest.approx(2 / 3)
        assert metrics.pos_match_ratio("кот", []) is None

    def test_cost_uses_per_million_rates(self):
        assert metrics.cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
        assert metrics.cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.0)

    def test_cost_of_unknown_model_is_zero(self):
        assert metrics.cost_usd("нет-такой-модели", 1_000_000, 1_000_000) == 0.0

    def test_percentile_edges(self):
        values = list(range(1, 11))
        assert metrics.percentile(values, 95) == 10
        assert metrics.percentile(values, 50) == 5
        assert metrics.percentile([], 95) == 0.0


class TestSharovParsing:
    """Регрессия: парсер читал колонку части речи вместо частоты,
    ValueError гасился через continue, и словарь молча оставался пустым."""

    def _manager_for(self, tmp_path, content, monkeypatch):
        import generator
        (tmp_path / "sharov.csv").write_text(content, encoding="utf-8")
        monkeypatch.setattr(generator, "DATA_DIR", tmp_path)
        manager = generator.WordListManager.__new__(generator.WordListManager)
        manager.freq_lists, manager.relative_lists, manager.sharov = {}, {}, {}
        manager._load_sharov()
        return manager

    def test_reads_frequency_column_not_part_of_speech(self, tmp_path, monkeypatch):
        manager = self._manager_for(tmp_path, monkeypatch=monkeypatch, content=(
            "Lemma\tPoS\tFreq(ipm)\tR\tD\tDoc\n"
            "стол\ts\t402.5\t100\t97\t32332\n"
            "книга\ts\t413.9\t100\t95\t31000\n"))
        assert manager.get_sharov_frequency("стол") == pytest.approx(402.5)
        assert manager.get_sharov_frequency("книга") == pytest.approx(413.9)

    def test_sums_frequency_across_parts_of_speech(self, tmp_path, monkeypatch):
        manager = self._manager_for(tmp_path, monkeypatch=monkeypatch, content=(
            "Lemma\tPoS\tFreq(ipm)\tR\tD\tDoc\n"
            "а\tconj\t8198.0\t100\t97\t32332\n"
            "а\tintj\t19.8\t99\t90\t757\n"))
        assert manager.get_sharov_frequency("а") == pytest.approx(8217.8)

    def test_unknown_word_is_zero(self, tmp_path, monkeypatch):
        manager = self._manager_for(tmp_path, monkeypatch=monkeypatch, content=(
            "Lemma\tPoS\tFreq(ipm)\n" "стол\ts\t402.5\n"))
        assert manager.get_sharov_frequency("суэссоя") == 0
