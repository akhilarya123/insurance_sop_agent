FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3.11 install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY data ./data

EXPOSE 8000

ENV SOP_HOST=0.0.0.0 \
    SOP_PORT=8000

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
