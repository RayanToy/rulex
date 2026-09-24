"""Модульные тесты чистых функций: без сети и без обращений к LLM."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import metrics  # noqa: E402

from app.core.security import hash_password, needs_rehash, verify_password  # noqa: E402
from app.services.assessment import calculate_level  # noqa: E402


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
        from app.services import wordlists
        (tmp_path / "sharov.csv").write_text(content, encoding="utf-8")
        monkeypatch.setattr(wordlists, "DATA_DIR", tmp_path)
        manager = wordlists.WordListManager.__new__(wordlists.WordListManager)
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


class TestContentQuality:
    """Метрики, добавленные после разбора выдачи локальных моделей:
    формально задание строилось, а показывать его было нельзя."""

    def test_detects_chinese_in_russian_definition(self):
        # Реальный случай из прогона qwen2.5:7b-instruct
        assert metrics.foreign_letters("Слово,用来骂人或侮辱人的词语")

    def test_detects_mixed_alphabets_inside_word(self):
        # "вeterаны": латинские e, t, e, r среди кириллицы
        assert metrics.foreign_letters("вeterаны в совете") == ["e", "r", "t"]

    def test_clean_russian_passes(self):
        assert metrics.foreign_letters("Крупный населённый пункт") == []

    def test_punctuation_and_digits_are_not_letters(self):
        assert metrics.foreign_letters("Толкование — 5 слов, и точка.") == []

    def test_finds_duplicate_distractor(self):
        assert metrics.duplicate_distractors(["узнать", "выяснить", "выяснить"]) == ["выяснить"]

    def test_duplicates_ignore_case_and_spaces(self):
        assert metrics.duplicate_distractors(["Стол", " стол "]) == ["стол"]

    def test_unique_distractors_pass(self):
        assert metrics.duplicate_distractors(["дом", "лес", "река"]) == []

    def test_empty_values_are_skipped(self):
        assert metrics.duplicate_distractors(["", None, "дом"]) == []


class TestDistractorParsing:
    """Разбор ответа модели в список дистракторов.

    Прежний разбор брал всё после первого двоеточия и делил по запятым.
    На модели, которая вместо списка пишет рассуждение, это попадало
    в середину прозы; повторы не отсеивались (у qwen3 — 21% заданий
    с двумя одинаковыми вариантами); часть речи проверялась только
    по первому разбору pymorphy3.
    """

    @staticmethod
    def parse(response, word):
        from app.services import generator
        gen = generator.QuestionGenerator.__new__(generator.QuestionGenerator)
        return gen._parse_distractors(response, word)

    def test_plain_comma_list(self):
        result = self.parse("человек, житель, племя, народ", "кроманьонец")
        assert result == ["человек", "житель", "племя"]

    def test_numbered_list(self):
        result = self.parse("1. человек\n2. житель\n3. племя", "кроманьонец")
        assert result == ["человек", "житель", "племя"]

    def test_preamble_is_stripped(self):
        result = self.parse("Вот слова: человек, житель, племя", "кроманьонец")
        assert result == ["человек", "житель", "племя"]

    def test_duplicates_are_dropped(self):
        result = self.parse(
            "промокнуть, пропитаться, промокнуть, пропитаться, увянуть", "вымокать")
        assert len(result) == len(set(result))
        assert "промокнуть" in result

    def test_prose_refusal_yields_nothing(self):
        """Модель спорит вместо ответа — лучше пусто, чем мусор из прозы."""
        prose = ('Уважаемый пользователь, отмечу проблему.\n\n'
                 'Слово "сровняться" — орфографическая ошибка, такого слова нет.')
        assert self.parse(prose, "сровняться") == []

    def test_target_word_is_excluded(self):
        result = self.parse("лес, дерево, поле, роща", "лес")
        assert "лес" not in result

    def test_homonym_noun_is_kept(self):
        """«род» pymorphy3 первым разбором считает глаголом, но это
        нормальное существительное и терять его нельзя."""
        result = self.parse("род, племя, народ", "кроманьонец")
        assert "род" in result
