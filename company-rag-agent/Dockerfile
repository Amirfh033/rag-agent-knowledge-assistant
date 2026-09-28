FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY wheels wheels

RUN pip install --no-cache-dir --no-index --find-links=./wheels -r requirements.txt

COPY app app
COPY web web
COPY eval eval
COPY tests tests
COPY data/docs data/docs

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
