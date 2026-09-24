FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RULEX_DATA_DIR=/app/var

WORKDIR /app

# Зависимости — отдельным слоем: он переиспользуется, пока requirements.txt не менялся.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /app/data — частотные словари, они внутри образа.
# /app/var  — изменяемые данные (SQLite), сюда монтируется том.
RUN mkdir -p /app/var \
    && adduser --disabled-password --gecos "" --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
