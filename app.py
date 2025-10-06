import sys
import asyncio

# Fix for Windows ProactorEventLoop socket errors
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import gradio as gr
import numpy as np
import cv2
import tempfile
import torch
from ultralytics import YOLO
from src.utils import draw_boxes, crop_detections, read_text_from_crop

# Path to your trained YOLO model
MODEL_PATH = "best.pt"
_model, _class_names, _device = None, None, None


def get_model():
    global _model, _class_names, _device
    if _model is None:
        # Auto-select GPU if available
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using device: {_device}")
        _model = YOLO(MODEL_PATH)
        _class_names = _model.names
    return _model, _class_names, _device


def detect_image(img: np.ndarray, conf: float, iou: float, return_crops: bool, run_ocr: bool):
    """Detect plates in an image, optionally crop and run OCR."""
    model, names, device = get_model()
    res = model.predict(img, conf=conf, iou=iou, verbose=False, device=device)[0]
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [], []

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    annotated = draw_boxes(img, boxes_xyxy, confs, cls, [names[i] for i in range(len(names))])
    crops = crop_detections(img, boxes_xyxy) if return_crops or run_ocr else []
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
    ocr_texts = []
    if run_ocr and crops:
        for c in crops:
            texts = read_text_from_crop(c)
            ocr_texts.append(", ".join(texts))

    return annotated, crops_rgb, ocr_texts


def detect_video(video_path, conf: float, iou: float):
    """Process entire video and return annotated video file path."""
    model, names, device = get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("⚠️ Could not open video.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output temp file
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=conf, iou=iou, verbose=False, device=device)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            annotated = frame
        else:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            annotated = draw_boxes(frame, boxes_xyxy, confs, cls, [names[i] for i in range(len(names))])
        out.write(annotated)

    cap.release()
    out.release()
    return out_path


# ---------------- Gradio UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px; background: #1f2937; color: white; border-radius: 12px;">
            <h1>🚗 License Plate Detector (YOLOv8 + OCR)</h1>
            <p>Upload an image or video to detect license plates and optionally extract text with OCR.</p>
        </div>
        """
    )

    with gr.Tab("🔹 Image Detection"):
        with gr.Row():
            with gr.Column(scale=1):
                in_image = gr.Image(type="numpy", label="Upload Image", height=400)
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
                iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
                return_crops = gr.Checkbox(False, label="Return Cropped Plates Only")
                run_ocr = gr.Checkbox(False, label="Run OCR on Crops")
                btn = gr.Button("🚀 Detect Plates", variant="primary")
            with gr.Column(scale=1):
                out_image = gr.Image(type="numpy", label="Detections", height=400)
                out_gallery = gr.Gallery(label="Detected License Plates", columns=4, height=200)
                ocr_output = gr.Dataframe(headers=["Plate Text"], label="OCR Results", datatype=["str"], interactive=False)

        btn.click(
            fn=detect_image,
            inputs=[in_image, conf, iou, return_crops, run_ocr],
            outputs=[out_image, out_gallery, ocr_output]
        )

    with gr.Tab("🎥 Video Detection"):
        in_video = gr.Video(label="Upload Video")
        conf_v = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
        iou_v = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU")
        btn_v = gr.Button("🎬 Process Video", variant="primary")
        out_video = gr.Video(label="Processed Video with Detections")

        btn_v.click(fn=detect_video, inputs=[in_video, conf_v, iou_v], outputs=[out_video])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
