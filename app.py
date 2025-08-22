from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary.uploader
from cloudinary_config import cloudinary
from datetime import datetime, timedelta, timezone

# === Firebase Initialization ===
if "FIREBASE_SERVICE_ACCOUNT" in os.environ:
    service_account_json = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
    cred = credentials.Certificate(service_account_json)
else:
    cred = credentials.Certificate("C:\\Users\\acer\\Downloads\\project-fira-9b6d9-firebase-adminsdk-fbsvc-178360e8ca.json")  # Local testing

firebase_admin.initialize_app(cred)
db = firestore.client()

# === Debug info to check Railway runtime ===
print("🚀 TensorFlow version:", tf.__version__)
print("🖥️  Available devices:", tf.config.list_physical_devices())

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

app = Flask(__name__)

# === CORS Configuration ===
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins in development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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


def save_report_to_firestore(photo_url, prediction_json, geotag_location=None, cause_of_fire=None, user_id=None, user_name=None):
    # Set timezone to UTC+8 (Asia/Manila)
    manila_tz = timezone(timedelta(hours=8))
    local_time = datetime.now(manila_tz)
    # Format: August 22 10:30 pm
    formatted_time = local_time.strftime("%B %d %I:%M %p").replace(" 0", " ").replace("AM", "am").replace("PM", "pm")
    doc_ref = db.collection("fire_reports").document()
    doc_ref.set({
        "timestamp": formatted_time,
        "photo_url": photo_url,
        "geotag_location": geotag_location,
        "cause_of_fire": cause_of_fire[:200] if cause_of_fire else None,
        "reporter": user_name,  # Real user name
        "reporterId": user_id,  # User UID for filtering
        **prediction_json
    })

@app.route('/')
def home():
    return "Fire Detection API is running! (Fire + Structure + Smoke detection active)"

@app.route('/predict', methods=['POST'])
def predict():
    global fire_model, structure_model, smoke_model
    
    # Lazy load models
    if fire_model is None:
        print("Loading fire detection model...")
        fire_model = load_model("fire_mobilenet_model.keras")
    if structure_model is None:
        print("Loading structure classification model...")
        structure_model = load_model("structure_material_classifier.keras")
    if smoke_model is None:
        print("Loading smoke detection model...")
        smoke_model = load_model("smoke_balanced_mobilenetv2_model_fixed_final.keras")

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    num_structures = request.form.get('number_of_structures_on_fire')
    geotag_location = request.form.get('geotag_location')  # NEW FIELD
    cause_of_fire = request.form.get('cause_of_fire')      # NEW FIELD
    user_id = request.form.get('user_id')      # NEW FIELD
    user_name = request.form.get('user_name')  # NEW FIELD
    
    try:
        num_structures = int(num_structures)
    except (TypeError, ValueError):
        num_structures = None

    try:
        # ==== 1️⃣ Upload image to Cloudinary ====
        upload_result = cloudinary.uploader.upload(file, folder="fire_reports")
        photo_url = upload_result.get("secure_url")

        # ==== 2️⃣ Prepare image for prediction ====
        file.stream.seek(0)  # reset pointer since Cloudinary read it
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Make predictions
        fire_pred = fire_model.predict(image, verbose=0)[0]
        fire_result = FIRE_CLASSES[np.argmax(fire_pred)]
        fire_confidence = float(np.max(fire_pred)) * 100

        structure_pred = structure_model.predict(image, verbose=0)[0]
        structure_result = STRUCTURE_CLASSES[np.argmax(structure_pred)]

        smoke_pred = smoke_model.predict(image, verbose=0)[0]
        smoke_result = SMOKE_CLASSES[np.argmax(smoke_pred)]
        smoke_confidence = float(np.max(smoke_pred)) * 100

        alarm_level = determine_alarm_level(num_structures)

        # ==== 3️⃣ Save to Firestore ====
        prediction_data = {
            'prediction': fire_result,
            'confidence': f"{fire_confidence:.2f}%",
            'structure': structure_result,
            'number_of_structures_on_fire': num_structures,
            'alarm_level': alarm_level,
            'smoke_intensity': smoke_result,
            'smoke_confidence': f"{smoke_confidence:.2f}%"
        }
        save_report_to_firestore(photo_url, prediction_data, geotag_location, cause_of_fire, user_id, user_name)  # UPDATED

        # ==== 4️⃣ Return response ====
        return jsonify({
            'photo_url': photo_url,
            'geotag_location': geotag_location,  # NEW FIELD
            'cause_of_fire': cause_of_fire,      # NEW FIELD
            **prediction_data
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/get_reports', methods=['GET'])
def get_reports():
    try:
        # Get all reports from Firestore
        reports_ref = db.collection("fire_reports")
        reports = reports_ref.stream()
        
        reports_list = []
        for report in reports:
            report_data = report.to_dict()
            report_data['id'] = report.id
            # Ensure all expected fields are present
            report_data.setdefault('photo_url', None)
            report_data.setdefault('geotag_location', None)
            report_data.setdefault('cause_of_fire', None)
            report_data.setdefault('prediction', None)
            report_data.setdefault('confidence', None)
            report_data.setdefault('structure', None)
            report_data.setdefault('number_of_structures_on_fire', None)
            report_data.setdefault('alarm_level', None)
            report_data.setdefault('smoke_intensity', None)
            report_data.setdefault('smoke_confidence', None)
            report_data.setdefault('reporter', None)
            report_data.setdefault('reporterId', None)
            report_data.setdefault('timestamp', None)
            reports_list.append(report_data)
        
        # Sort by timestamp (newest first)
        reports_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify(reports_list)
    except Exception as e:
        print(f"Error retrieving reports: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/update_report/<report_id>', methods=['PUT'])
def update_report(report_id):
    try:
        data = request.get_json()
        
        # Get the report document
        report_ref = db.collection("fire_reports").document(report_id)
        report_doc = report_ref.get()
        
        if not report_doc.exists:
            return jsonify({'error': 'Report not found'}), 404
        
        # Update only allowed fields
        update_data = {}
        if 'cause_of_fire' in data:
            update_data['cause_of_fire'] = data['cause_of_fire'][:200]  # Limit to 200 chars
        
        if 'number_of_structures_on_fire' in data:
            try:
                num_structures = int(data['number_of_structures_on_fire'])
                update_data['number_of_structures_on_fire'] = num_structures
                # Recalculate alarm level
                update_data['alarm_level'] = determine_alarm_level(num_structures)
            except (TypeError, ValueError):
                update_data['number_of_structures_on_fire'] = None
                update_data['alarm_level'] = "Unknown - structure count not provided"
        
        # Update the document
        report_ref.update(update_data)
        
        # Return updated report
        updated_doc = report_ref.get()
        return jsonify(updated_doc.to_dict())
        
    except Exception as e:
        print(f"Error updating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_report/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    try:
        # Get the report document
        report_ref = db.collection("fire_reports").document(report_id)
        report_doc = report_ref.get()
        
        if not report_doc.exists:
            return jsonify({'error': 'Report not found'}), 404
        
        # Delete the document
        report_ref.delete()
        
        return jsonify({'message': 'Report deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting report: {str(e)}")
        return jsonify({'error': str(e)}), 500 

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)