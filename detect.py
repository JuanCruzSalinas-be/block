from ultralytics import YOLO
import cv2
import torch

VIDEO_PATH = "output.ts"

# More accurate than nano
model = YOLO("yolo26s.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

# Use Apple GPU when available
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",

        conf=0.08,
        imgsz=640,
        max_det=500,
        classes=VEHICLE_CLASSES,

        device=device,

        verbose=False
    )

    result = results[0]
    vehicle_count = 0

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

        vehicle_count = len(track_ids)

        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences
        ):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Vehicle {track_id} {confidence:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Tracked Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()