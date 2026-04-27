FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Sentence-BERT model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Copy project structure
COPY CODE/ ./CODE/
COPY DATA/  ./DATA/

# Output directories
RUN mkdir -p EVALUATIONS

# Default command — run from repo root so OUT_DIR resolves correctly
CMD ["python", "CODE/main.py", "--source", "synthetic", "--n", "48", "--k", "6", "--topics", "6"]
