"""Страница и статика после выноса CSS и JS из шаблона.

Главный риск такого выноса — разрыв между разметкой и скриптом:
обработчик onclick="fn()" остался в HTML, а функция переименована или
пропала. Браузер об этом молчит до первого клика, поэтому связь
проверяется здесь по исходникам.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")

HANDLER = re.compile(r"""\bon[a-z]+=\\?["']\s*([A-Za-z_$][\w$]*)\(""")


def defined_functions(js: str) -> set[str]:
    return set(re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", js, re.M))


@pytest.mark.parametrize("name", ["templates/index.html", "static/js/app.js"])
def test_every_inline_handler_is_defined(name):
    # в app.js разметка с обработчиками собирается из строк
    handlers = set(HANDLER.findall(TEMPLATE if name.endswith(".html") else SCRIPT))
    assert handlers, f"в {name} не найдено ни одного обработчика — регулярка сломалась?"
    missing = handlers - defined_functions(SCRIPT)
    assert not missing, f"{name}: обработчики без функции в app.js: {sorted(missing)}"


# Текстовые поля, которые пишут модель, преподаватель или сам пользователь.
# Числа и значения-перечисления (id, score, level) сюда не входят.
FREE_TEXT = re.compile(r"\b(question|target_word|definition|correct_answer|distractor_\d|"
                       r"username|full_name|email|message|detail|option|word)\b")


def interpolations(js: str) -> list[str]:
    """Выражения внутри ${…} с учётом вложенных скобок."""
    found, start = [], js.find("${")
    while start != -1:
        depth, i = 1, start + 2
        while depth:
            depth += {"{": 1, "}": -1}.get(js[i], 0)
            i += 1
        found.append(js[start + 2:i - 1].strip())
        start = js.find("${", i)
    return found


def test_free_text_is_escaped_before_insertion():
    """Толкования и дистракторы пишет модель или преподаватель, а видит их
    каждый ученик. Вставленный в разметку без экранирования текст вида
    <img src=x onerror=…> исполнялся бы в браузере ученика."""
    exprs = interpolations(SCRIPT)
    assert len(exprs) > 30, "интерполяции не найдены — разбор сломался?"
    unescaped = [e for e in exprs if FREE_TEXT.search(e) and not e.startswith("escapeHtml(")]
    assert not unescaped, f"текст без escapeHtml: {unescaped}"


def test_no_inline_code_left_in_template():
    assert "<style" not in TEMPLATE
    assert re.search(r"<script(?![^>]*\bsrc=)[^>]*>", TEMPLATE) is None


@pytest.mark.parametrize("path, content_type", [("/static/css/app.css", "text/css"),
                                                ("/static/js/app.js", "javascript")])
def test_page_links_versioned_assets(client, path, content_type):
    page = client.get("/")
    assert page.status_code == 200
    match = re.search(re.escape(path) + r"\?v=([0-9a-f]{12})\b", page.text)
    assert match, f"страница не ссылается на {path} с версией"

    asset = client.get(match.group(0))
    assert asset.status_code == 200
    assert content_type in asset.headers["content-type"]
