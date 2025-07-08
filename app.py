from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model("fire_cnn_model.keras")  # Make sure this file is in the same folder

CLASSES = ['No Fire', 'Fire']

@app.route('/')
def home():
    return "🔥 Fire Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = image.resize((128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        'prediction': CLASSES[class_index],
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
