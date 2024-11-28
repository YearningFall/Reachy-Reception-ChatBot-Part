# Reachy-Reception

This project is a combination of two parts, the facial detection-recognition, and the chatbot system.
This is the chatbot component of the Reception system for Reachy, a humanoid robot. The chatbot part is designed to facilitate interactive conversations with users, enhancing customer service experiences. 
Even though this project is programmed with Reachy Robot in mind, this project could be applied anywhere as long as you have appropriate hardware and after adjusting a few parameters in existing codes.
The successful operation of this chatbot requires face detection integration to recognize users before initiating the chatbot functionality.

## Reception Project

This project aims to create an interactive, chatbot-based receptionist system using the Reachy robot. The system is intended for customer service environments and leverages various technologies for seamless and efficient human-robot interaction:
- **Vosk Model** for lightweight speech recognition
- **Ollama** as a language model to generate context-aware responses
- **Pico TTS** for producing spoken responses
- **Flask Server** for handling data communication between Reachy and the local machine
- **YOLOv5** for the facial detection process
- **dlib library** for the facial recognition process and for facial embedding database

## Project Overview

The Reception Chatbot Project enables the Reachy robot to function as a receptionist by:
1. **Listening for and recognizing specific keywords** via the lightweight Vosk speech-to-text model.
2. **Sending confirmed text data to Ollama** on a local machine for generating responsive answers.
3. **Vocalizing responses using Pico TTS**, enabling natural and efficient voice interaction.
4. **Managing real-time communication** between the robot and the local system through Flask.
5. **Face detection and recognition** to let the robot detect and recognize between guests and staff.

## Key Components

- **Vosk**: A lightweight speech-to-text (STT) engine for recognizing keywords in spoken input, ideal for low-resource devices like the Raspberry Pi 3.
- **Flask**: A micro web framework to handle HTTP requests for data exchange between Reachy and the local machine.
- **Ollama**: A large language model hosted on a local machine that generates responses based on user input.
- **Pico TTS**: Text-to-speech engine used for delivering spoken responses, optimized for lightweight and clear audio on a Raspberry Pi 3.
- **YOLOv5**: An object detection model, used to find faces before cropping it to be sent to facial recognition. 
- **dlib**: A C++ toolkit used for its facial recognition, face embeddings received from dlib can be saved into .pkl files.

## Project Structure

- **app**: Contains the Flask server files and API endpoints.
- **models**: Contains files and configurations for Vosk, Ollama, Pico TTS, and YOLOv5 weight.
- **scripts**: Helper scripts for setup, deployment, and configuration.

## Installation

### Prerequisites
Before setting up the project, ensure the following dependencies are installed:

- **Python 3.8+**
- **Flask**: Install Flask via pip for handling HTTP requests.
    ```bash
    pip install flask
    ```
1. **Vosk Model**
   - Download from: [Vosk Small Model (English)](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)
   - Extract and configure the path in the project files.

2. **Ollama**
   - Install via: `curl -fsSL https://ollama.com/install.sh | sh`
   - Pull the `llama3.1` model: `ollama pull llama3.1`.

3. **Pico TTS**
   - Install using: `sudo apt install libttspico-utils`.
   - Test with: `pico2wave -w test.wav "Hello, this is a test." && aplay test.wav`.

4. **YOLOv5**
   - Install via: `git clone https://github.com/ultralytics/yolov5.git && cd yolov5 && pip install -r requirements.txt`
   - For more information, refer to https://github.com/ultralytics/yolov5

5. **dlib**
   - Install via: `pip install dlib`
---

### Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/reception-chatbot.git
    cd reception-chatbot
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Flask**:
   - Update the IP addresses in the configuration files to match your local and server setup.

4. **Start the Flask Server**:
    ```bash
    python app/server.py
    ```

5. **Run the Chatbot System**:
   - Ensure Reachy, the Flask server, and the local machine are connected. Then initialize the chatbot system.

---

## Usage

The Reception Chatbot follows these steps during operation:

1. **Face Detection**:
   - The robot must first detect a face, before cropping the face and sending it to be recognized.

2. **Face Recognition**:
   - dlib checks, if the face detected, is similar to any faces stored in the embedding.

4. **Keyword Detection**: 
   - The Reachy robot listens for specific keywords, such as "Reception," using the Vosk model.
   
5. **Confirmation Phase**:
   - The system enters a confirmation phase, where the user confirms detected speech. Confirmed inputs are sent to the Flask server for processing.

6. **Ollama Response Generation**:
   - The local system, hosting Ollama, processes the confirmed input to generate context-aware responses.

7. **Response Delivery**:
   - The response is sent back to Reachy, which vocalizes it using Pico TTS, completing the interaction loop.

---


## License

This project is licensed under the **MIT License**. Feel free to use and modify it as per your needs.

---

Let me know if you need further refinements or additional sections for your README!
