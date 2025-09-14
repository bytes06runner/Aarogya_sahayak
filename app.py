
import configparser
import logging
import threading
import time
import json
import requests
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from geopy.distance import geodesic
import uuid
from datetime import datetime


class SmartChunkManager:
    """ Manages the creation and storage of 'SmartChunks' of data. """
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        # Ensure the directory for the chunk database exists
        if not os.path.exists(os.path.dirname(self.storage_path)):
            os.makedirs(os.path.dirname(self.storage_path))

    def create_smart_chunk(self, user_query, phone_number, environmental_data, language, location):
        """Creates a structured dictionary for a single interaction."""
        return {
            "chunk_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.utcnow().isoformat(),
            "user_phone_hash": hash(phone_number), # Simple hash for anonymization
            "user_query": user_query,
            "detected_language": language,
            "location": location,
            "environmental_data": environmental_data,
            "prediction": None # To be filled in after model inference
        }

    def save_chunk(self, chunk):
        """Appends a chunk to the local JSONL database file."""
        try:
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(chunk) + '\n')
            logging.info(f"Saved chunk {chunk['chunk_id']} to local storage.")
        except IOError as e:
            logging.error(f"Could not save chunk to {self.storage_path}: {e}")

class ModelInference:
    """ Handles loading the TFLite model and making predictions. """
    def __init__(self, model_path: str, labels_path: str):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            logging.info("TFLite model and labels loaded successfully.")
        except Exception as e:
            logging.error(f"CRITICAL: Failed to load TFLite model or labels: {e}")
            raise

    def predict(self, chunk):
        """Runs inference on a chunk of data."""
        try:
            text_input = np.array([chunk['user_query']], dtype=np.string_)
            num_input = np.array([[
                float(chunk['environmental_data'].get('aqi', 100)),
                float(chunk['environmental_data'].get('temperature_c', 30.0))
            ]], dtype=np.float32)

            text_input_index = next((d['index'] for d in self.input_details if 'text' in d['name']), None)
            num_input_index = next((d['index'] for d in self.input_details if 'numerical' in d['name']), None)
            
            if text_input_index is None or num_input_index is None:
                raise RuntimeError("Could not find text or numerical input tensors in the model.")

            self.interpreter.set_tensor(text_input_index, text_input)
            self.interpreter.set_tensor(num_input_index, num_input)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction_index = np.argmax(output_data)
            risk_score = float(np.max(output_data))
            
            condition = self.labels[str(prediction_index)]
            
            advice_code = f"{condition}_HIGH_RISK" if risk_score > 0.7 else f"{condition}_MEDIUM_RISK"
            if "CRITICAL" in condition:
                advice_code = condition

            return {"advice_code": advice_code, "risk_score": f"{risk_score:.0%}"}
        except Exception as e:
            logging.error(f"Error during model prediction: {e}", exc_info=True)
            return {"advice_code": "GENERAL_WELLNESS_MEDIUM_RISK", "risk_score": "N/A"}

class NotificationHandler:
    """ Manages sending SMS/WhatsApp messages via the Twilio API. """
    def __init__(self, account_sid: str, auth_token: str, twilio_number: str):
        if not all([account_sid, auth_token, twilio_number]) or 'YOUR_ACCOUNT_SID' in account_sid:
            logging.warning("Twilio is not configured. SIMULATING message sending.")
            self.client = None
        else:
            try:
                self.client = Client(account_sid, auth_token)
                logging.info("Twilio client initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Twilio client: {e}")
                self.client = None
        self.twilio_number = twilio_number

    def send_message(self, recipient_number: str, message_body: str):
        if not self.client:
            logging.warning(f"SIMULATING SEND to {recipient_number}: '{message_body}'")
            return
        try:
            message = self.client.messages.create(
                from_=self.twilio_number,
                body=message_body,
                to=recipient_number
            )
            logging.info(f"Message sent to {recipient_number}. SID: {message.sid}")
        except Exception as e:
            logging.error(f"Failed to send Twilio message: {e}")

class LanguageSupport:
    """ Manages language detection and translation of advice. """
    def __init__(self):
        self.ADVICE_TEMPLATES = {
            "HEART_ATTACK_CRITICAL": "CRITICAL HEALTH ALERT! Chest pain and shortness of breath can be signs of a heart attack. Please seek immediate medical attention or call an ambulance. An emergency alert has been sent to nearby services.",
            "DENGUE_FEVER_HIGH_RISK": "High risk of Dengue Fever detected. Symptoms include high fever, severe headache, and joint pain. Rest and drink plenty of fluids. See a doctor immediately if you experience stomach pain or bleeding.",
            "COVID_19_HIGH_RISK": "Symptoms match COVID-19. Please isolate yourself immediately to prevent spreading. Monitor your oxygen levels and consult a doctor for testing and treatment options.",
            "PREGNANCY_CONCERN_HIGH_RISK": "We've detected a potential pregnancy-related concern. It is very important to consult your gynecologist or the nearest hospital as soon as possible for a checkup.",
            "MALARIA_HIGH_RISK": "Symptoms are consistent with Malaria. This includes fever, chills, and sweating. It is crucial to get a blood test and see a doctor for proper medication.",
            "JAUNDICE_HIGH_RISK": "Yellowing of skin and eyes suggests Jaundice. This requires medical attention to determine the cause. Please see a doctor.",
            "CHICKENPOX_MEDIUM_RISK": "Itchy rash and blisters are common with Chickenpox. Keep the skin clean, avoid scratching, and rest. Consult a doctor if the fever is very high or if there are signs of infection.",
            "GENERAL_WELLNESS_MEDIUM_RISK": "Thank you for checking in. Remember to stay hydrated and maintain a balanced diet. If you feel unwell, please describe your symptoms in more detail.",
        }

    def detect_language(self, text):
        return "en"

    def get_translated_advice(self, advice_code, risk_score, language="en"):
        base_message = self.ADVICE_TEMPLATES.get(advice_code, self.ADVICE_TEMPLATES["GENERAL_WELLNESS_MEDIUM_RISK"])
        return (
            f"**Aarogya Sahayak - Health Advisory**\n\n"
            f"*Condition Alert:* {advice_code.replace('_', ' ').title()}\n"
            f"*Risk Level:* {risk_score}\n\n"
            f"*Advice:* {base_message}\n\n"
            f"_Disclaimer: This is an AI-generated suggestion. Please consult a doctor for a medical diagnosis._"
        )

class EmergencyHandler:
    """ Handles the emergency protocol for critical predictions. """
    def __init__(self, hospitals_path: str, notification_handler: NotificationHandler):
        self.notification_handler = notification_handler
        try:
            with open(hospitals_path, 'r') as f:
                self.hospitals = json.load(f)
            logging.info(f"Loaded {len(self.hospitals)} hospitals from {hospitals_path}")
        except Exception as e:
            logging.error(f"Could not load hospitals file '{hospitals_path}': {e}")
            self.hospitals = []

    def trigger_emergency_protocol(self, chunk):
        logging.critical(f"EMERGENCY PROTOCOL TRIGGERED for user {chunk['user_phone_hash']}")
        emergency_contact = "+919999988888"
        user_location = (chunk['location']['latitude'], chunk['location']['longitude'])
        nearest_hospital = self.hospitals[0] if self.hospitals else {"name": "the nearest hospital", "phone": "whatsapp:+911122334455"}
        alert_message = (
            f"URGENT AI Health Alert: Potential critical condition "
            f"({chunk['prediction']['advice_code']}) detected for a user. "
            f"Location: http://maps.google.com/maps?q={user_location[0]},{user_location[1]}. "
            f"Case ID: {chunk['chunk_id']}"
        )
        self.notification_handler.send_message(nearest_hospital['phone'], alert_message)
        self.notification_handler.send_message(f"whatsapp:{emergency_contact}", alert_message)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
config = configparser.ConfigParser()
config.read('config.ini')

try:
    TWILIO_ACCOUNT_SID = config.get('twilio', 'account_sid')
    TWILIO_AUTH_TOKEN = config.get('twilio', 'auth_token')
    TWILIO_NUMBER = config.get('twilio', 'twilio_number')
    MODEL_PATH = config.get('app', 'model_path')
    LABELS_PATH = config.get('app', 'labels_path')
    CHUNK_DB_PATH = config.get('app', 'chunk_database_path')
    HOSPITALS_PATH = config.get('app', 'hospitals_path')
    SYNC_INTERVAL_SECONDS = config.getint('app', 'sync_interval_seconds')
except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logging.error(f"CRITICAL: Configuration Error in config.ini: {e}. App cannot start.")
    exit(1)

logging.info("Initializing application components...")
chunk_manager = SmartChunkManager(storage_path=CHUNK_DB_PATH)
model_inference_engine = ModelInference(model_path=MODEL_PATH, labels_path=LABELS_PATH)
notification_handler = NotificationHandler(
    account_sid=TWILIO_ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    twilio_number=TWILIO_NUMBER
)
language_translator = LanguageSupport()
emergency_manager = EmergencyHandler(
    hospitals_path=HOSPITALS_PATH,
    notification_handler=notification_handler
)
logging.info("All components initialized.")

@app.route('/chat', methods=['POST'])
def chat_webhook():
    try:
        user_message = request.values.get('Body', '').strip()
        sender_id = request.values.get('From', '')
        logging.info(f"Received message from {sender_id}: '{user_message}'")
        if not user_message or not sender_id:
            return Response(status=200)
        
        env_data = {"aqi": 150, "temperature_c": 35.0, "source": "default"}
        location = {"latitude": 22.87, "longitude": 88.41}
        lang = "en"

        chunk = chunk_manager.create_smart_chunk(
            user_query=user_message, phone_number=sender_id,
            environmental_data=env_data, language=lang, location=location
        )
        prediction = model_inference_engine.predict(chunk)
        chunk['prediction'] = prediction
        advice_message = language_translator.get_translated_advice(
            advice_code=prediction['advice_code'],
            risk_score=prediction['risk_score'],
            language=lang
        )
        notification_handler.send_message(sender_id, advice_message)
        if "CRITICAL" in prediction['advice_code']:
            emergency_manager.trigger_emergency_protocol(chunk)
        chunk_manager.save_chunk(chunk)
        return Response(status=200)
    except Exception as e:
        logging.error(f"Error in chat webhook: {e}", exc_info=True)
        return Response(status=500)

@app.route('/health', methods=['GET'])
def health_check():
    return "Aarogya Sahayak is running!", 200

if __name__ == '__main__':
    logging.info("Starting Flask web server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5001, debug=False)


