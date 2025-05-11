## 🐳 2. Dockerfile (обновлённый и исправленный)

# Используем официальный образ Python
FROM python:3.11-slim-bookworm

# Рабочая директория
WORKDIR /app

# Копируем зависимости и файлы
COPY requirements.txt .

# Устанавливаем пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY app/ ./app/
COPY abalone_regression_dashboard.yaml .
COPY regression_explainer.joblib .

RUN python dashboard.py

# Экспортируем порт
EXPOSE 8050

# Запускаем приложение
ENTRYPOINT ["python"]
CMD ["app.py"]