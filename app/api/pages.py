"""HTML-страница приложения."""
import hashlib

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import PROJECT_ROOT

router = APIRouter(include_in_schema=False)

# Путь от корня проекта, а не от текущего каталога: раньше приложение
# находило шаблоны, только если его запускали из корня репозитория.
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
STATIC_DIR = PROJECT_ROOT / "static"


def asset_version() -> str:
    """Хеш содержимого CSS и JS для ссылок вида app.js?v=….

    Без него браузер после деплоя может взять из кэша старый скрипт,
    который обращается к API в прежнем формате. Считается один раз при
    старте: статика меняется только вместе с деплоем.
    """
    digest = hashlib.sha256()
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.suffix in {".css", ".js"}:
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


ASSET_VERSION = asset_version()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"asset_version": ASSET_VERSION})
