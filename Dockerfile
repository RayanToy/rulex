FROM python:3.11-slim

# Публичный экземпляр: первый зарегистрированный не становится
# администратором — иначе им оказался бы случайный посетитель.
# Кука сессии только по HTTPS (за прокси Amvera — так и есть).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RULEX_DATA_DIR=/app/var \
    RULEX_FIRST_USER_ADMIN=0 \
    RULEX_COOKIE_SECURE=1

WORKDIR /app

# Зависимости — отдельным слоем: он переиспользуется, пока requirements.txt не менялся.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /app/data — частотные словари и готовый банк вопросов, они внутри образа.
# /app/var  — изменяемые данные (SQLite), сюда монтируется постоянное хранилище.
RUN mkdir -p /app/var \
    && adduser --disabled-password --gecos "" --uid 1000 app \
    && chown -R app:app /app \
    && chmod +x /app/docker-entrypoint.sh

# Пользователь app включается в точке входа — после того как смонтированному
# хранилищу выданы права (см. docker-entrypoint.sh)
ENTRYPOINT ["/app/docker-entrypoint.sh"]

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
