# Use official TensorFlow image (TensorFlow already installed)
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install only the additional packages we need (very fast)
RUN pip install --no-cache-dir flask pillow

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE $PORT

# Run the application
CMD ["python", "app.py"]