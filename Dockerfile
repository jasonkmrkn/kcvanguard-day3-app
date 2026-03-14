# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/home/user/.local/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production - Final runtime image
# =============================================================================
FROM python:3.10-slim

RUN useradd -m -u 1000 user

USER user

ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY --chown=1000:1000 . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]