import cv2
import numpy as np
import pickle
import base64
import face_recognition
import requests
import time
import ollama  # Importing Ollama for interaction

# Server URLs for Reachy
server_url = 'http://192.168.3.92:5001/send_image'  # Replace with Reachy's IP
recognition_url = 'http://192.168.3.92:5001/receive_recognition'  # Endpoint to send recognition result
ollama_response_url = 'http://192.168.3.92:5001/receive_ollama_response'  # Endpoint to send Ollama's response to Reachy
speech_url = 'http://192.168.3.92:5001/send_speech'  # Endpoint to receive speech from Reachy

# Load known face encodings and names
try:
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings = pickle.load(f)

    with open('face_names.pkl', 'rb') as f:
        known_face_names = pickle.load(f)
except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
    exit(1)
except Exception as e:
    print(f"Error loading encodings or names: {e}")
    exit(1)

# Function to interact with Ollama and generate a response
def send_to_ollama(text):
    ollama_response = ollama.chat(model='Reception', messages=[{"role": "user", "content": text}])
    
    response_content = ollama_response.get("message", {}).get("content", "No response found")
    
    print(f"Ollama's content: {response_content}")  # Debug
    return response_content

def get_ollama_welcome(name):
    if name == "Guest":
        message = "Say welcome guest"
    else:
        message = f"Say welcome {name}"
    
    print(f"Sending to Ollama: {message}")  # Debug
    return send_to_ollama(message)

def process_image(image_string):
    if image_string:
        print("Received encoded image string from Reachy.")  # Log for received string
    else:
        print("No image string received.")  # Log in case image string is missing
    
    image_bytes = base64.b64decode(image_string)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if decoded_image is not None:
        print("Image successfully decoded.")  # Log for successful decoding
        face_locations = face_recognition.face_locations(decoded_image)
        face_encodings = face_recognition.face_encodings(decoded_image, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Guest"  # Default to "Guest"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        if face_names:
            print(f"Recognized face: {face_names[0]}")  # Log recognized face name
        else:
            print("No recognizable faces detected.")  # Log if no face recognized

        # Send recognized name back to Reachy
        payload = {'recognized_name': face_names[0] if face_names else "Guest"}
        try:
            requests.post(recognition_url, json=payload)
            print(f"Sent recognized name: {payload['recognized_name']} to Reachy")  # Log recognized name sent
        except Exception as e:
            print(f"Failed to send recognized name to Reachy: {e}")

        return face_names[0] if face_names else "Guest"
    else:
        print("Failed to decode the image.")  # Log if image decoding failed
        return None

# Function to send Ollama's response back to Reachy
def send_ollama_response_to_reachy(response_message):
    payload = {'ollama_response': response_message}
    try:
        response = requests.post(ollama_response_url, json=payload)
        if response.status_code == 200:
            print(f"Sent Ollama response to Reachy: {response_message}")
        else:
            print(f"Failed to send Ollama response to Reachy. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending Ollama response to Reachy: {e}")

# Function to handle incoming speech data from Reachy
def process_speech(speech_data):
    if speech_data:
        print(f"Processing received speech: {speech_data}")
        # Send speech input to Ollama for processing
        ollama_response = send_to_ollama(speech_data)
        # Send Ollama's response back to Reachy
        send_ollama_response_to_reachy(ollama_response)
    else:
        print("No speech data received.")

def main():
    while True:
        try:
            # Continuously check for new images from Reachy
            print("Checking for images from Reachy...")
            response = requests.get(server_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                image_string = data.get('string_image')

                if image_string:
                    print(f"Received encoded image string: {image_string[:30]}...")  # Log part of the encoded string
                    recognized_name = process_image(image_string)
                    
                    # Send the recognized name to Ollama for a response every time a face is detected
                    welcome_message = get_ollama_welcome(recognized_name)
                    print(welcome_message)
                        
                    # Send Ollama's response back to Reachy
                    send_ollama_response_to_reachy(welcome_message)

                else:
                    print("No image string found in the server's response.")
            else:
                print(f"Failed to retrieve data from server. Status code: {response.status_code}")
        
            # Continuously check for speech data from Reachy
            print("Checking for speech data from Reachy...")
            speech_response = requests.get(speech_url, timeout=10)
            if speech_response.status_code == 200:
                speech_data = speech_response.json().get('speech_data', '')
                process_speech(speech_data)
            else:
                print(f"No speech data found. Status code: {speech_response.status_code}")
        
        except requests.exceptions.RequestException as req_error:
            print(f"Request failed: {req_error}")

        # Add a small delay to prevent overwhelming the system
        time.sleep(1)

if __name__ == '__main__':
    main()
