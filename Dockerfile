# Use Python base image
FROM python:3.9

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Render
EXPOSE 8080

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
