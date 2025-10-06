title: Numberplatedetector
emoji: 🐨
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: false


# 🚗 License Plate Detector (YOLOv8 + OCR)

This is a Gradio web app for detecting license plates in **images and videos** using a YOLOv8 model, with optional OCR (EasyOCR) to read text from plates.

---

## 🔹 Features
- Upload an **image** → get detections + cropped plates + OCR results
- Upload a **video** → get an annotated MP4 with bounding boxes
- Trained YOLOv8 model (`best_lp_yolov8.pt`) included

---

## 🔹 Run Locally

```bash
pip install -r requirements.txt
python app.py
