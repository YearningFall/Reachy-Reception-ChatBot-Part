import cv2
import time

def record_video(duration=6, output_file='output.avi'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0  # Adjust based on your system and camera

    # Try a reliable codec like 'MJPG' or 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Change to 'XVID' if MJPG doesn't work
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not open video file for writing.")
        return

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)  # Write the frame to the video file
            cv2.imshow('Recording...', frame)
            cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video(duration=6, output_file='output.avi')
