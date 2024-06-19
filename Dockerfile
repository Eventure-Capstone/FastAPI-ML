# Gunakan image dasar resmi dari Python
FROM python:3.9-slim

# Set environment variable untuk tidak menulis bytecode Python dan untuk port
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies untuk membangun aplikasi
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Buat direktori untuk aplikasi
WORKDIR /app

# Copy requirements.txt untuk menginstal dependencies
COPY requirements.txt /app/

# Buat virtual environment dan instal dependencies di dalamnya
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Tambahkan virtual environment ke PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy seluruh kode aplikasi ke dalam Docker container
COPY . /app

# Set environment variable untuk port
ENV PORT 8080
EXPOSE 8080

# Jalankan aplikasi FastAPI dengan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
