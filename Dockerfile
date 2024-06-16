# Gunakan image base Python
FROM python:3.9-slim

# Set lingkungan kerja
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode ke dalam container
COPY . .

# Expose port aplikasi
EXPOSE 8080

# Tentukan perintah untuk menjalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
