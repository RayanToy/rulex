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
