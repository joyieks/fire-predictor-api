FROM tensorflow/tensorflow:2.19.0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE $PORT
CMD ["python", "app.py"]
