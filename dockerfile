FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLflow model (CRITICAL)
COPY model/ /app/model/

# Copy application code
COPY src/ /app/src/

EXPOSE 7860

CMD ["python", "-m", "src.app.main"]
