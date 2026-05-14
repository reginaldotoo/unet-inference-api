FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY UNet_Inference.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]