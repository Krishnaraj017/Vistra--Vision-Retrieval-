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
    # Additional system utilities
    libmagic1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install requirements
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . .

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]