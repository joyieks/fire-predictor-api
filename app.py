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

# ✅ Load models once
fire_model = load_model("fire_mobilenet_model.keras")
structure_model = load_model("structure_classifier_model_finetuned.keras")
smoke_model = load_model("final_smoke_model.keras")

FIRE_CLASSES = ['Fire', 'No Fire']
STRUCTURE_CLASSES = ['Concrete Building', 'Metal Structure', 'Wooden Houses']
SMOKE_CLASSES = ['high', 'low', 'medium']  # Based on sorted_val folder structure

# 🚨 Helper: Determine alarm level based on number of structures on fire
def determine_alarm_level(count):
    if count is None:
        return "Unknown - structure count not provided"
    if count >= 80:
        return "General Alarm - Major area affected (80 fire trucks)"
    elif count >= 36:
        return "Task Force Delta - 36 fire trucks"
    elif count >= 32:
        return "Task Force Charlie - 32 fire trucks"
    elif count >= 28:
        return "Task Force Bravo - 28 fire trucks"
    elif count >= 24:
        return "Task Force Alpha - 24 fire trucks"
    elif count >= 20:
        return "Fifth Alarm - 20 fire trucks"
    elif count >= 16:
        return "Fourth Alarm - 16 fire trucks"
    elif count >= 12:
        return "Third Alarm - 12 fire trucks"
    elif count >= 8:
        return "Second Alarm - 8 fire trucks"
    elif count >= 4:
        return "First Alarm - 4 fire trucks"
    elif count >= 1:
        return "Under Control - Low fire risk"
    else:
        return "Fireout - Fire has been neutralized"

@app.route('/')
def home():
    return "🔥 Fire Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Optional input from frontend/Postman
    num_structures = request.form.get('number_of_structures_on_fire')
    try:
        num_structures = int(num_structures)
    except (TypeError, ValueError):
        num_structures = None

    try:
        # 🔍 Preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # 🔥 Fire prediction
        fire_pred = fire_model.predict(image, verbose=0)[0]
        fire_class_index = np.argmax(fire_pred)
        fire_result = FIRE_CLASSES[fire_class_index]
        fire_confidence = float(np.max(fire_pred)) * 100

        # 🏗️ Structure prediction
        structure_pred = structure_model.predict(image, verbose=0)[0]
        structure_result = STRUCTURE_CLASSES[np.argmax(structure_pred)]

        # 🌫️ Smoke intensity prediction
        smoke_pred = smoke_model.predict(image, verbose=0)[0]
        smoke_result = SMOKE_CLASSES[np.argmax(smoke_pred)]

        # 🚨 Alarm level
        alarm_level = determine_alarm_level(num_structures)

        # ✅ Return result
        return jsonify({
            'prediction': fire_result,
            'confidence': f"{fire_confidence:.2f}%",
            'structure': structure_result,
            'number of structures on fire': num_structures,
            'alarm level': alarm_level,
            'smoke intensity': smoke_result,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
