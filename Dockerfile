# Используем официальный образ Python
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт 8000
EXPOSE 8000

# Запускаем сервер uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]