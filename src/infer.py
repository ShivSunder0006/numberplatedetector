import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from src.utils import draw_boxes, save_result, crop_detections

def run_inference(
    weights: str,
    source: str,
    out_dir: str = "outputs",
    conf: float = 0.25,
    iou: float = 0.45,
    save_crops: bool = False,
):
    model = YOLO(weights)
    names = model.names

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # YOLO handles images, dirs, videos seamlessly via .predict, but we’ll manually post-process to draw with our util
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        verbose=False,
        stream=True  # iterate results for large folders or videos
    )

    # If it's an image or directory of images, this loop writes annotated images.
    # For video, YOLO returns frames; we’ll accumulate via cv2.VideoWriter when we detect a video file.
    is_video = str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    vw = None

    for idx, r in enumerate(results):
        # r.orig_img is BGR numpy array
        orig = r.orig_img
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            annotated = orig
            boxes_xyxy = np.empty((0, 4))
            confs = np.array([])
            cls = np.array([])
        else:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            annotated = draw_boxes(orig, boxes_xyxy, confs, cls, [names[i] for i in range(len(names))])

        if is_video:
            if vw is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = out_dir / (Path(source).stem + "_pred.mp4")
                vw = cv2.VideoWriter(str(out_path), fourcc, 30, (w, h))
            vw.write(annotated)
        else:
            out_path = out_dir / f"{Path(getattr(r.path,'name',f'frame_{idx}')).stem}_pred.jpg"
            save_result(annotated, out_path)

        if save_crops and len(boxes_xyxy) > 0:
            crops = crop_detections(orig, boxes_xyxy)
            for i, c in enumerate(crops):
                crop_path = out_dir / "crops" / f"{Path(getattr(r.path,'name',f'frame_{idx}')).stem}_lp_{i}.jpg"
                save_result(c, crop_path)

    if vw is not None:
        vw.release()
        print("Saved video with predictions.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    ap.add_argument("--source", type=str, required=True, help="Image/dir/video path")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save_crops", action="store_true")
    args = ap.parse_args()

    run_inference(**vars(args))
