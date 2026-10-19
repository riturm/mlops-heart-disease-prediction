# ---- Dockerfile : builds the model-serving API (Task 6) ----
FROM python:3.11-slim

WORKDIR /app

# Install deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, serving app and the trained model.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# Serve with uvicorn. (--host 0.0.0.0 so it is reachable from outside the container.)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
