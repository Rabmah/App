import cv2
import numpy as np
from sort import Sort
from yolov5 import YOLOv5

# Initialize YOLOv5 model
yolo = YOLOv5("yolov5s.pt", device='cpu')

# Initialize SORT tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

video_path = "../Videos/people-6387.mp4"
cap = cv2.VideoCapture(video_path)

# Get video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define fixed frame size
fixed_width = 1000
fixed_height = 650

# Calculate resize scale
scale_width = fixed_width / frame_width
scale_height = fixed_height / frame_height
scale = min(scale_width, scale_height)

# Counter and line parameters
car_count = 0
person_count = 0
counted_cars = set()
counted_people = set()

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Calculate line position based on frame height
    line_position = int(frame.shape[0] * 0.6)

    # Perform object detection with YOLOv5
    results = yolo.predict(frame)
    detections = results.xyxy[0]

    car_detections = []
    person_detections = []

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        if int(cls) == 2:  # Class ID for 'car' in COCO dataset
            car_detections.append([x1, y1, x2, y2, conf.item()])
        elif int(cls) == 0:  # Class ID for 'person' in COCO dataset
            person_detections.append([x1, y1, x2, y2, conf.item()])

    tracked_cars = tracker.update(np.array(car_detections)) if len(car_detections) > 0 else []
    tracked_people = tracker.update(np.array(person_detections)) if len(person_detections) > 0 else []


    def update_count(tracked_objects, counted_objects, line_position):
        new_count = 0
        for x1, y1, x2, y2, obj_id in tracked_objects:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            center_y = int((y1 + y2) / 2)

            if obj_id not in counted_objects:
                if center_y <= line_position:
                    new_count += 1
                    counted_objects.add(obj_id)

                # Draw frame around detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            else:
                # Draw frame around tracked object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return new_count


    car_count += update_count(tracked_cars, counted_cars, line_position)
    person_count += update_count(tracked_people, counted_people, line_position)

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)
    cv2.putText(frame, f"Cars passed: {car_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"People passed: {person_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()