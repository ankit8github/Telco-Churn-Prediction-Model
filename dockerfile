# 1. Use official lightweight Python image
FROM python:3.11-slim

# 2. Set working directory (src is our module root)
WORKDIR /app/src

# 3. Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file first (better layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# 5. Copy application source code
COPY src/ /app/src/

# 6. Copy bundled inference model (tracked in git)
COPY src/serving/model /app/src/serving/model

# 7. Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 8. Expose Gradio / FastAPI port
EXPOSE 7860

# 9. Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
