# Use a modern Python 3.11 base
FROM python:3.11-slim

# Avoid running as root in container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps needed for building wheels (faiss, etc.)
# Keep the image small by cleaning apt lists
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git libffi-dev wget && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python deps early to leverage Docker cache
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Copy app code
COPY . .

# Create a low-privilege user and use it
RUN useradd -m appuser || true
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Run Uvicorn (single-process is fine for a portfolio/demo; Render will manage scaling)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
