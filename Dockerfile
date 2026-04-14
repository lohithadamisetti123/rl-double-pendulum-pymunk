FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for pygame, X11, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV PIP_DEFAULT_TIMEOUT=300

# Upgrade pip to a modern version
RUN pip install --no-cache-dir --upgrade pip

# Install all Python deps from requirements (torch included)
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["bash"]