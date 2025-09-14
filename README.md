Aarogya Sahayak - AI Health Advisor
This is the main application for the Aarogya Sahayak chatbot. It's a Flask-based server designed to be run on an Android device using Termux. It uses a pre-trained TensorFlow Lite model to provide offline, AI-powered preventive health advice via SMS and WhatsApp.

Project Structure
SIH-Final-Project/
├── aarogya_sahayak/      # Main application package
│   └── ... (python modules)
├── data/
│   └── hospitals.json
├── models/               # Contains the trained AI model
│   ├── model.tflite
│   └── labels.json
├── training_utils/       # Contains the scripts to train the AI
│   ├── health_data.csv
│   └── train_model.py
├── venv/                 # Your Python virtual environment
├── chatbot_main.py       # Main script to run the server
├── config.ini            # Your secret keys and config
└── requirements.txt      # Python packages to install

Setup and Run Instructions
1. Create and Activate a Virtual Environment:
(Make sure you are in the SIH-Final-Project directory)

# Deactivate the training venv if it's still active
deactivate

# Create a new venv for the application
python3 -m venv venv_app
source venv_app/bin/activate

2. Install Dependencies:

python3 -m pip install -r requirements.txt

3. Configure the Application:

Rename config.ini.example to config.ini.

Open config.ini and fill in your Twilio Account SID, Auth Token, and Twilio phone number.

Rename data/hospitals.json.example to data/hospitals.json and add real local hospital data if possible.

4. Run the Server:

python3 chatbot_main.py

The server will start on port 5000. You will need to use a tool like ngrok to expose this port to the internet so Twilio can send messages to it.

