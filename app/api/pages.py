"""HTML-страница приложения."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import PROJECT_ROOT

router = APIRouter(include_in_schema=False)

# Путь от корня проекта, а не от текущего каталога: раньше приложение
# находило шаблоны, только если его запускали из корня репозитория.
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
