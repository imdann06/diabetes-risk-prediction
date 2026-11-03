FROM python:3.11-slim

WORKDIR /app

# Copiar requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . /app

# Mantener la imagen ligera
ENV PYTHONUNBUFFERED=1

CMD ["python", "predict.py", "--help"]
