from flask import Flask, render_template, Response
import cv2
import torch
import pygame

app = Flask(__name__)

# Load the YOLOv5 model with error handling
try:
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=r'C:/Users/bhara/PycharmProjects/ABD_FLASK/best.pt')
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Define your list of class names based on your YOLOv5 model
class_names = ['CAP', 'COAT', 'GLOVE']  # Replace with your actual class names

# Initialize a flag to track whether an alarm should be triggered
trigger_alarm = False

# Initialize pygame for audio playback
pygame.mixer.init()

# Load your audio file (replace 'your_audio_file.mp3' with your actual audio file)
audio_file_path = 'C:/Users/bhara/PycharmProjects/ABD_FLASK/alert.mp3'
pygame.mixer.music.load(audio_file_path)

def play_alarm():
    pygame.mixer.music.play()

def generate_frames():
    global trigger_alarm
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        results = model(frame)

        # Extract the detected objects and their labels
        detected_objects = results.xyxy[0].cpu().numpy()

        # Process and draw detections
        if len(detected_objects) == 0:
            # No objects detected, trigger the alarm and capture a picture
            trigger_alarm = True
            cv2.imwrite('C:/Users/bhara/PycharmProjects/ABD_FLASK/no_objects_detected.jpg', frame)
            # Play the alarm audio
            play_alarm()
        else:
            trigger_alarm = False

            for obj in detected_objects:
                x1, y1, x2, y2, confidence, class_id = obj

                if confidence > 0.5:  # Adjust this threshold as needed
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Draw bounding box and label on the frame
                    color = (0, 255, 0)  # BGR color for the box (in this case, green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Get the class name from the list based on the class_id
                    class_name = class_names[int(class_id)]
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
