from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from collections import defaultdict
import threading
#Grab a single image from the webcam
#detects object in the frame
#Create a temp dictionary to hold max confidence per class
app = Flask(__name__)
model = YOLO("Ishpreet.pt")

target_classes = ['Laptop', 'Bottle', 'Mouse', 'Watch', 'Ishpreet']
detected_confidences = defaultdict(float)

cap = cv2.VideoCapture(0)

lock = threading.Lock()

def gen_frames():
    global detected_confidences
    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)[0]  # ✅ This disables all logging


        temp_confidences = defaultdict(float)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name in target_classes and conf > 0.5:
                if conf > temp_confidences[class_name]:
                    temp_confidences[class_name] = conf

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name}: {conf:.2f}"
                #Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,0,0), 2)

        with lock:
            detected_confidences = {cls: round(temp_confidences.get(cls, 0.0), 2) for cls in target_classes}

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in HTTP multipart format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    with lock:
        return jsonify(detected_confidences)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

    #sql server database architecture diagram