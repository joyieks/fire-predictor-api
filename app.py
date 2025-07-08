from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

# 🧠 Prevent TensorFlow from allocating all memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass  # Fail silently if not supported

app = Flask(__name__)

# ✅ Load the model only once
model = load_model("fire_cnn_model.keras")
CLASSES = ['No Fire', 'Fire']

@app.route('/')
def home():
    return "🔥 Fire Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        # 🔍 Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((128, 128))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # 🔮 Predict
        prediction = model.predict(image, verbose=0)[0]
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'prediction': CLASSES[class_index],
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 👇 For local development; ignored on Render
    app.run(debug=False, host='0.0.0.0', port=10000)
