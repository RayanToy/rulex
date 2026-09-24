"""Направление зависимостей между слоями.

api -> services -> core, и никогда обратно. Сервисы и ядро не знают про
HTTP: их можно вызывать из скриптов, eval-харнесса и тестов без FastAPI.
Нарушение этого правила не ломает ни одного теста поведения, поэтому
оно проверяется отдельно — по импортам в исходниках.
"""
import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

FORBIDDEN = {
    "core": ("app.api", "app.services", "fastapi", "starlette"),
    "services": ("app.api", "fastapi", "starlette"),
}


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_upwards(layer):
    violations = [
        f"{path.relative_to(APP.parent)}: {module}"
        for path in sorted((APP / layer).glob("*.py"))
        for module in sorted(imported_modules(path))
        if module.startswith(FORBIDDEN[layer])
    ]
    assert not violations, "импорт в обратную сторону:\n" + "\n".join(violations)
