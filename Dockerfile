# Use official TensorFlow image (TensorFlow already installed)
FROM tensorflow/tensorflow:2.15.0-py3

# Set working directory
WORKDIR /app

# Install only the additional packages we need (very fast)
RUN pip install --no-cache-dir flask gunicorn pillow

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE $PORT

# Run the application
CMD ["python", "app.py"]