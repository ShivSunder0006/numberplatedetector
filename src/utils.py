import easyocr
import cv2

_ocr_reader = easyocr.Reader(["en"], gpu=False)  # use gpu=True if CUDA available

def draw_boxes(image, boxes_xyxy, confidences, class_ids, class_names):
    img = image.copy()
    for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy.astype(int), confidences, class_ids.astype(int)):
        label = f"{class_names[int(cid)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def crop_detections(image, boxes_xyxy):
    crops = []
    h, w = image.shape[:2]
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crops.append(image[y1:y2, x1:x2].copy())
    return crops

def read_text_from_crop(crop):
    results = _ocr_reader.readtext(crop)
    texts = [text for (_, text, conf) in results if conf > 0.4]
    return texts if texts else ["<unreadable>"]
