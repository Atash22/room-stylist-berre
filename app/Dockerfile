FROM python:3.10-slim

WORKDIR /app

# System deps for Pillow/torch image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Weights file (style_classifier_effnet_b0.pt) should be copied into this
# directory before building, or mounted as a volume at runtime.

CMD ["python", "app.py"]
