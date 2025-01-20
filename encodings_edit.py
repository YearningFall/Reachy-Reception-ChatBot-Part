import pickle
import cv2
import os
import face_recognition
import time
import numpy as np

VIDEO_DURATION = 6  # in seconds
FRAME_INTERVAL = 5 # Process every nth frame

def load_encodings():
    global face_encodings, face_names
    try:
    # Load the encodings and names from the files
        with open('face_encodings.pkl', 'rb') as f:
            face_encodings = pickle.load(f)
        with open('face_names.pkl', 'rb') as f:
            face_names = pickle.load(f)
        # Now `face_encodings` is a list of face embeddings (numpy arrays),
        # and `face_names` is a list of corresponding names.
    except FileNotFoundError:
        print("Encoding files not found. Creating empty lists.")
        return [], []  # Return empty lists if files don't exist
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return [], []  # Return empty lists if there's another error

def view_list():
    global face_encodings, face_names
    unique_names = set(face_names)
    name_counts = {name: face_names.count(name) for name in unique_names}
    try:
        clear_screen()
        print("Unique names and number of encodings per person:")
        for name, count in name_counts.items():
            print(f"{name}: {count} encoding(s)")
            # Optional: Print total number of encodings
        print(f"Total number of encodings: {len(face_encodings)}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return 

def delete_data(encodings_file='face_encodings.pkl', names_file='face_names.pkl'):
    global face_encodings, face_names
    try:
        print("Deletion of data cannot be undone. Are you sure? \nType 'q' to cancel")
        name_to_delete = input("Enter the name of the person to delete: ")
        if name_to_delete == 'q':
            print("Deletion cancelled")
            return
        
        updated_encodings = []
        updated_names = []
        # Remove the specified person's encodings and names
        for encoding, name in zip(face_encodings, face_names):
            if name != name_to_delete:
                updated_encodings.append(encoding)
                updated_names.append(name)

        # Save updated encodings and names back to the files
        with open(encodings_file, 'wb') as f_encodings:
            pickle.dump(updated_encodings, f_encodings)
        with open(names_file, 'wb') as f_names:
            pickle.dump(updated_names, f_names)
    
        print(f"{name_to_delete} has been deleted from the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return 

def trim_embeddings(encodings_file='face_encodings.pkl', names_file='face_names.pkl', max_embeddings=100):
    global face_encodings, face_names
    try:
        print("Trimming of data cannot be undone. Are you sure? \nType 'q' to cancel")
        name_to_trim = input("Enter the name of the person whose data you would like to trim: ")
        
        if name_to_trim == 'q':
            print("Trimming cancelled")
            return
        # Get indices of all embeddings corresponding to the specified person
        index_to_keep = [i for i, n in enumerate(face_names) if n == name_to_trim]
        print(f"Found {len(index_to_keep)} encodings for {name_to_trim}.")
        # If the number of embeddings exceeds max limit, trim the oldest ones
        if len(index_to_keep) > max_embeddings:
            # Trim older entries, keeping only the most recent max_embeddings
            index_to_remove = index_to_keep[:-max_embeddings]
            print(f"Trimming {len(index_to_remove)} old embeddings.")
        
            updated_encodings = [enc for i, enc in enumerate(face_encodings) if i not in index_to_remove]
            updated_names = [n for i, n in enumerate(face_names) if i not in index_to_remove]
            # Update global variables
            face_encodings = updated_encodings
            face_names = updated_names
            
            # Save updated encodings and names back to the files
            with open(encodings_file, 'wb') as f_encodings:
                pickle.dump(updated_encodings, f_encodings)
            with open(names_file, 'wb') as f_names:
                pickle.dump(updated_names, f_names)
            print(f"Trimmed embeddings for {name_to_trim}. Kept the most recent {max_embeddings} embeddings.")
        else:
            print(f"No trimming needed. {name_to_trim} has only {len(index_to_keep)} embeddings.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return 

def add_embeddings(encodings_file='face_encodings.pkl', names_file='face_names.pkl'):
    global face_encodings, face_names
    try:
        new_person_name = input("Enter the name of the new person: ")
        if new_person_name == 'q':
            return
        video_file = record_video()  # Step 1: Record video
        if video_file:
            face_encodings = extract_frames_and_encode(video_file)  # Step 2: Extract frames and encode
            if face_encodings:
                save_encodings_to_file(new_person_name, face_encodings)  # Step 3: Save encodings to file
    except Exception as e:
        print(f"An error occurred: {e}")
    return

# Function to record video for a specified duration
def record_video(duration=VIDEO_DURATION):
    cap = cv2.VideoCapture(0)  # Open the default camera (webcam)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Set video parameters
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second
    frame_width = int(cap.get(3))  # Width of the frames
    frame_height = int(cap.get(4))  # Height of the frames
    
    # Define the codec and create VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
    
    print("Recording video for {} seconds...".format(duration))
    start_time = time.time()
    
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Write the frame into the video file
            cv2.imshow('Recording...', frame)  # Show the frame in real-time
            
            # Stop the recording if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print("Video recording completed.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 'output.avi'  # Return the recorded video file name

# Extract frames from the video for face encoding
def extract_frames_and_encode(video_path, frame_interval=FRAME_INTERVAL):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    
    frame_count = 0
    face_encodings = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Check the image shape and type
            print(f"Frame shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
            # Ensure the image is in RGB format
            if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 3:
                angles = [-15, 0, 15]
                for angle in angles:
                    rotated_frame = rotate_image(rgb_frame, angle)
                    # Get face encodings for the frame
                    encodings = face_recognition.face_encodings(rotated_frame)
                    if encodings:
                        # Add any found encodings to our list
                        # Assuming there's only one face per frame
                        face_encodings.append(encodings[0])
            else:
                print("Error: Frame is not in RGB format.")
            
        frame_count += 1

    cap.release()
    
    print(f"Extracted {len(face_encodings)} face encodings from the video.")
    return face_encodings

def rotate_image(image, angle):
    if image.ndim != 3 and image.shape[2] != 3:
        print(f"Error: Image is not in RGB format. Shape: {image.shape}")
        return image 
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    rotated = cv2.convertScaleAbs(rotated)
    print("Rotated image")
    print(f"Rotated Image: dtype={rotated.dtype}, shape={rotated.shape}, ndim={rotated.ndim}")
    
    return rotated

# Save face encodings and names to file
def save_encodings_to_file(name, encodings, encodings_file='face_encodings.pkl', names_file='face_names.pkl'):
    # Load existing encodings
    try:
        with open(encodings_file, 'rb') as f_encodings:
            all_encodings = pickle.load(f_encodings)
    except FileNotFoundError:
        all_encodings = []
    
    try:
        with open(names_file, 'rb') as f_names:
            all_names = pickle.load(f_names)
    except FileNotFoundError:
        all_names = []
    
    # Append new encodings and name
    all_encodings.extend(encodings)
    all_names.extend([name] * len(encodings))
    
    # Save updated encodings and names back to the files
    with open(encodings_file, 'wb') as f_encodings:
        pickle.dump(all_encodings, f_encodings)
    with open(names_file, 'wb') as f_names:
        pickle.dump(all_names, f_names)
    
    print(f"Added {len(encodings)} encodings for {name} to the database.")

# Function to recognize faces from the webcam
def webcam_face_recognition():
    global face_encodings, face_names
    # Open the webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame from BGR (OpenCV) to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        detected_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_locations) == 0:
            print("No face detected in this frame.")
            continue

        # Initialize an array for the recognized face names
        detected_face_names = []

        for detected_face_encoding in detected_face_encodings:
            # Compare the current face with known faces
            matches = face_recognition.compare_faces(face_encodings, detected_face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance if a match is found
            if matches:
                face_distances = face_recognition.face_distance(face_encodings, detected_face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = face_names[best_match_index]

            detected_face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, detected_face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

        # Hit 'q' on the keyboard to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

def photo_face_recognition():
    global face_encodings, face_names
    photo_path = 'wasin.jpg'
    print(f"Recognizing faces in the photo: {photo_path}")
    
    # Load the image
    image = cv2.imread(photo_path)
    
    if image is None:
        print("Error: Could not load the image.")
        return
    
    # Convert the image to grayscale and then back to RGB
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    detected_face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    print(f"Face Locations: {face_locations}")
    print(f"Detected Face Encodings: {detected_face_encodings}")

    # Skip if no faces are detected
    if len(face_locations) == 0:
        print("No faces detected in the image.")
        return

    # Initialize a list to hold names
    detected_face_names = []

    for detected_face_encoding in detected_face_encodings:
        matches = face_recognition.compare_faces(face_encodings, detected_face_encoding)
        name = "Unknown"  # Default name

        if matches:
            face_distances = face_recognition.face_distance(face_encodings, detected_face_encoding)
            best_match_index = np.argmin(face_distances)

            # Check if the best match is valid
            if matches[best_match_index]:
                name = face_names[best_match_index]

        detected_face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, detected_face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the image with recognized faces
    cv2.imshow('Face Recognition on Photo', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    try:
        load_encodings()
        print("============== * * * ==============")
        print("Main Menu:")
        print("1: View list of known faces")
        print("2: Trim data")
        print("3: Add data")
        print("4: Delete data")
        print("5: Test facial recognition")
        print("6: Photo face recognition")
        print("q: Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            view_list()
            main()

        elif choice == '2':
            trim_embeddings()
            main()

        elif choice == '3':
            add_embeddings()
            main()

        elif choice == '4':
            delete_data()
            main()

        elif choice == '5':
            webcam_face_recognition()
            main()

        elif choice == '6':
            photo_face_recognition()
            main()    

        elif choice == 'q':
            print("Exiting program...")

        else:
            print("Invalid choice. Please try again.")
            main()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()