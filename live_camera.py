import cv2
import torch
from ultralytics import YOLO
from src.utils import draw_boxes, crop_detections, read_text_from_crop

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Using device:", device)
model = YOLO("best.pt")

def run_live_camera(conf=0.25, iou=0.45, run_ocr=False):
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("⚠️ Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        res = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)[0]
        boxes = res.boxes

        if boxes is not None and len(boxes) > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            # Draw bounding boxes
            frame = draw_boxes(frame, boxes_xyxy, confs, cls, [model.names[i] for i in range(len(model.names))])

            # OCR
            if run_ocr:
                crops = crop_detections(frame, boxes_xyxy)
                for i, c in enumerate(crops):
                    texts = read_text_from_crop(c)
                    cv2.putText(
                        frame,
                        f"OCR: {', '.join(texts)}",
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # Show live detections
        cv2.imshow("📡 Live License Plate Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_camera(conf=0.25, iou=0.45, run_ocr=True)
