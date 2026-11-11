FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUCLEI_MODEL_DIR=/app/model \
    PCAM_MODEL_DIR=/app/pcam_model

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

EXPOSE 8000
CMD ["python", "-m", "app.start"]

