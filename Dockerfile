FROM python:3.11-slim    
# Install system dependencies for OpenCV, graphics, and unstructured
RUN apt-get update && apt-get install -y \
    # Graphics and OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    # PDF processing dependencies
    poppler-utils \
    tesseract-ocr \
    # Image processing dependencies
    libheif-dev \
    # Additional system utilities
    libmagic1 \
    gcc \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Upgrade pip and install system-level dependencies
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    # Specific installations for problematic packages
    torch \
    opencv-python-headless \
    # Install rest of the requirements
    -r requirements.txt

# Copy application code
COPY . .

# Optional: Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
