from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import os


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

app = Flask(__name__)

# Model variables (loaded on demand)
fire_model = None
structure_model = None
smoke_model = None

FIRE_CLASSES = ['Fire', 'No Fire']
STRUCTURE_CLASSES = ['Concrete Building', 'Metal Structure', 'Wooden Houses']
SMOKE_CLASSES = ['high', 'low', 'medium']

def determine_alarm_level(count):
    if count is None:
        return "Unknown - structure count not provided"
    if count >= 80: return "General Alarm - Major area affected (80 fire trucks)"
    elif count >= 36: return "Task Force Delta - 36 fire trucks"
    elif count >= 32: return "Task Force Charlie - 32 fire trucks"
    elif count >= 28: return "Task Force Bravo - 28 fire trucks"
    elif count >= 24: return "Task Force Alpha - 24 fire trucks"
    elif count >= 20: return "Fifth Alarm - 20 fire trucks"
    elif count >= 16: return "Fourth Alarm - 16 fire trucks"
    elif count >= 12: return "Third Alarm - 12 fire trucks"
    elif count >= 8:  return "Second Alarm - 8 fire trucks"
    elif count >= 4:  return "First Alarm - 4 fire trucks"
    elif count >= 1:  return "Under Control - Low fire risk"
    else:             return "Fireout - Fire has been neutralized"

@app.route('/')
def home():
    return "Fire Detection API is running! (Fire + Structure + Smoke detection active)"

@app.route('/predict', methods=['POST'])
def predict():
    global fire_model, structure_model, smoke_model

    # Lazy load models
    if fire_model is None:
        fire_model = load_model("fire_mobilenet_model.keras")
    if structure_model is None:
        structure_model = load_model("structure_material_classifier.keras")
    if smoke_model is None:
        smoke_model = load_model("smoke_balanced_mobilenetv2_model.keras")

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    num_structures = request.form.get('number_of_structures_on_fire')
    try:
        num_structures = int(num_structures)
    except (TypeError, ValueError):
        num_structures = None

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        fire_pred = fire_model.predict(image, verbose=0)[0]
        fire_result = FIRE_CLASSES[np.argmax(fire_pred)]
        fire_confidence = float(np.max(fire_pred)) * 100

        structure_pred = structure_model.predict(image, verbose=0)[0]
        structure_result = STRUCTURE_CLASSES[np.argmax(structure_pred)]

        smoke_pred = smoke_model.predict(image, verbose=0)[0]
        smoke_result = SMOKE_CLASSES[np.argmax(smoke_pred)]
        smoke_confidence = float(np.max(smoke_pred)) * 100

        alarm_level = determine_alarm_level(num_structures)

        return jsonify({
            'prediction': fire_result,
            'confidence': f"{fire_confidence:.2f}%",
            'structure': structure_result,
            'number_of_structures_on_fire': num_structures,
            'alarm_level': alarm_level,
            'smoke_intensity': smoke_result,
            'smoke_confidence': f"{smoke_confidence:.2f}%",
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)