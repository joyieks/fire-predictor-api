# Use official TensorFlow image (TensorFlow already installed)
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Copy requirements before installing
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE $PORT

# Run the application
CMD ["python", "app.py"]
