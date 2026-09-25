"""Морфология через pymorphy3: часть речи, лемма, совпадение части речи."""
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

_POS_NAMES = {"NOUN": "существительное", "VERB": "глагол", "INFN": "инфинитив"}


def get_pos(word: str) -> str | None:
    parsed = morph.parse(word)
    if not parsed:
        return None
    return parsed[0].tag.POS


def get_lemma(word: str) -> str:
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].normal_form
    return word.lower()


def pos_name(pos: str | None) -> str:
    """Часть речи по-русски — для промптов."""
    return _POS_NAMES.get(pos, "существительное")


def pos_matches(candidate: str, target_pos: str | None) -> bool:
    """Совпадает ли часть речи хотя бы по одному разбору.

    pymorphy3 возвращает разборы по убыванию вероятности, и у
    омонимов верный не всегда первый: «род» разбирается сначала как
    глагольная форма, и нормальное существительное отбрасывалось.
    """
    if target_pos is None:
        return False
    for parse in morph.parse(candidate):
        pos = parse.tag.POS
        if pos == target_pos:
            return True
        # INFN и VERB — одна часть речи для целей теста
        if {pos, target_pos} <= {"VERB", "INFN"}:
            return True
    return False
