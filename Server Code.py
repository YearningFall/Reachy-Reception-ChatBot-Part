import time
import cv2
import torch
import base64
import numpy as np
import requests
from flask import Flask, request, jsonify
import threading
import os  # For espeak
from vosk import Model, KaldiRecognizer  # For voice recognition (Vosk)
import pyaudio  # For capturing microphone input

app = Flask(__name__)

# Global variables
string_image = None
speech_data = None  # Global variable to store the speech data
speech_sent = False  # Flag to track if speech has already been sent
lock = threading.Lock()  # To ensure thread safety

# Load YOLOv5 model (lightweight model suitable for Raspberry Pi 3)
model_path = 'model/colab_yolov5m_r320_b16_e318.pt'
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)

# Load Vosk model for speech detection (Reception keyword)
vosk_model = Model("/home/pi/Reception/vosk-model-small-en-us-0.15")  # Updated path to vosk model in /home/pi/Reception
recognizer = KaldiRecognizer(vosk_model, 16000)  # Initialize recognizer for English
mic = pyaudio.PyAudio()

@app.route('/receive_image', methods=['POST'])
def receive_image():
    global string_image
    data = request.get_json()
    with lock:
        string_image = data.get('string_image')
    print("Received image data from local.")  # Debugging log for image receipt
    return jsonify({'message': 'Image received successfully'}), 200

@app.route('/send_image', methods=['GET'])
def send_image():
    global string_image
    with lock:
        if string_image:
            print("Sending encoded image string to local.")  # Debug log
            return jsonify({'string_image': string_image}), 200
    print("No image available to send.")  # Debug log if no image is available
    return jsonify({'message': 'No image available'}), 404

@app.route('/send_speech', methods=['GET'])
def send_speech():
    global speech_data, speech_sent, string_image
    string_image = None  # Reset the image after sending speech
    if speech_data and not speech_sent:  # Only send speech data once
        speech_sent = True  # Mark the speech as sent
        response_data = {'speech_data': speech_data}
        speech_data = None  # Clear the speech data after sending
        print(f"Sending speech data to local: {response_data}")  # Debug log
        return jsonify(response_data), 200
    print("No speech data available or already sent.")  # Debug log
    return jsonify({'message': 'No speech data available or already sent'}), 404

@app.route('/receive_ollama_response', methods=['POST'])
def receive_ollama_response():
    data = request.get_json()
    ollama_response = data.get('ollama_response', "No response received")
    print(f"Received Ollama response: {ollama_response}")
    
    # Speak the response using espeak
    try:
        os.system(f'espeak "{ollama_response}"')
        print(f"Spoken: {ollama_response}")
        
        # Restart listening for "Reception" after speaking
        wait_for_reception_trigger()
    except Exception as e:
        print(f"Failed to speak the text: {e}")
    
    return jsonify({'message': 'Response received and spoken'}), 200

# Function to start the Flask server
def start_flask_server(port=5001):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# Start the Flask server in a separate thread on port 5001
flask_thread = threading.Thread(target=start_flask_server, args=(5001,))
flask_thread.daemon = True  # Daemon thread will exit when the main program exits
flask_thread.start()

def detect_and_send_face():
    # Face detection with webcam
    cap = cv2.VideoCapture(2)  # Changed to use camera device 2
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detection_interval = 2  # Set detection interval (in seconds)
    last_detection_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                try:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_frame = cv2.resize(gray_frame, (320, 320))  # Match model's expected input size

                    # Run YOLOv5 inference to detect a face
                    results = model(resized_frame)
                    bboxes = results.xyxy[0].cpu().numpy()

                    if len(bboxes) > 0:
                        print("Face detected. Stopping detection loop.")
                        
                        for bbox in bboxes:
                            x1, y1, x2, y2, conf, cls = map(int, bbox[:6])
                            cropped_face = resized_frame[y1:y2, x1:x2]

                            _, buffer = cv2.imencode('.jpg', cropped_face)
                            image_bytes = base64.b64encode(buffer)
                            string_image = image_bytes.decode('utf-8')

                            print(f"Encoded image string created: {string_image[:30]}...")  # Log part of encoded string

                        # Send the image to the MacBook (local server)
                        server_url = 'http://192.168.3.92:5001/receive_image'
                        payload = {'string_image': string_image}
                        try:
                            response = requests.post(server_url, json=payload)
                            if response.status_code == 200:
                                print(f"Image sent successfully: {response.json()}")
                                cap.release()
                                cv2.destroyAllWindows()
                                break  # Stop detection loop
                            else:
                                print(f"Failed to send image. Status code: {response.status_code}")
                        except Exception as e:
                            print(f"Error sending image: {e}")
                    else:
                        print("No face detected, continuing loop...")

                    last_detection_time = current_time
                except Exception as e:
                    print(f"Error during face detection: {e}")

    except KeyboardInterrupt:
        print("Task stopped by user (Ctrl+C).")

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

    print("Camera closed. Flask server is still running.")

# Vosk speech detection function for "Reception"
def listen_for_reception():
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    print("Microphone is open. Listening for 'Reception'...")  # Added status message
    
    # Initialize start_time here
    start_time = time.time()  # Initialize start_time properly

    while True:
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            if "reception" in result.lower():
                print("Word 'Reception' detected.")
                stream.stop_stream()
                return True
        else:
            if time.time() - start_time > 10:  # Timeout set to 10 seconds for Reception word
                print("Reception not detected. Returning to face detection.")
                stream.stop_stream()
                return False

# Wait for "Reception" for 10 seconds
def wait_for_reception_trigger():
    if listen_for_reception():
        # Make Reachy speak "I am listening"
        print("Saying: 'I am listening'")
        os.system('espeak "I am listening"')  # Call espeak to speak "I am listening"
        print("Listening for user speech for 5 seconds.")
        capture_speech()
    else:
        # If "Reception" is not detected, return to face detection
        print("Restarting face detection after no 'Reception' detected.")
        detect_and_send_face()  # Restart face detection

# Capture speech after "Reception" is heard
def capture_speech():
    global speech_data, speech_sent
    # Reset the speech sent flag to False so we can send new data
    speech_sent = False
    # Capture user's speech after "Reception" detected and send to MacBook
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    print("Microphone is open. Listening for user speech...")  # Added status message for speech capture

    speech_data = ""
    start_time = time.time()
    while time.time() - start_time < 5:  # Wait 5 seconds for user speech
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            speech_data += result

    # Continue listening for 2 seconds of silence after speech is detected
    silent_start = time.time()
    while time.time() - silent_start < 2:
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            speech_data += result
            silent_start = time.time()  # Reset silence timer if speech is detected

    stream.stop_stream()

    if speech_data:
        print(f"Captured speech: {speech_data}")
        confirm_speech()
    else:
        print("No speech detected. Returning to face detection.")
        detect_and_send_face()

# Function to confirm the detected speech
def confirm_speech():
    global speech_data
    # Parse the actual text from the detected speech
    try:
        parsed_result = eval(speech_data)
        detected_text = parsed_result.get("text", "")
    except Exception as e:
        print(f"Error parsing speech data: {e}")
        detected_text = speech_data  # Fallback if parsing fails
    
    if detected_text:
        # Speak the captured speech and ask for confirmation
        os.system(f'espeak "You said {detected_text}. Is it correct?"')
        print(f"Asking for confirmation: 'You said {detected_text}. Is it correct?'")

        # Listen for yes/no confirmation
        if listen_for_confirmation():
            print("Speech confirmed as correct.")
            # Send speech to local server if confirmed
            send_speech_to_local(detected_text)
        else:
            print("Speech not confirmed. Restarting speech capture.")
            capture_speech()
    else:
        print("No valid speech detected. Returning to face detection.")
        detect_and_send_face()

# Function to listen for yes/no confirmation
def listen_for_confirmation():
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    print("Listening for 'yes' or 'no'...")  # Added status message for confirmation

    start_time = time.time()
    while time.time() - start_time < 5:  # 5 seconds to capture confirmation
        data = stream.read(4096)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            if "yes" in result.lower():
                print("User confirmed with 'yes'.")
                stream.stop_stream()
                return True
            elif "no" in result.lower():
                print("User denied with 'no'.")
                stream.stop_stream()
                return False

    stream.stop_stream()
    print("No confirmation detected. Restarting speech capture.")
    return False

# Send the captured speech to the local server
def send_speech_to_local(speech_text):
    global speech_data, speech_sent
    # Send captured speech to MacBook
    server_url = 'http://192.168.3.92:5001/receive_speech'
    payload = {'speech_data': speech_text}
    try:
        response = requests.post(server_url, json=payload)
        if response.status_code == 200:
            print("Speech data sent to local successfully.")
            speech_sent = True
        else:
            print(f"Failed to send speech data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending speech data to local: {e}")

# Start the face detection process
detect_and_send_face()

# The Flask server keeps running
while True:
    time.sleep(1)
