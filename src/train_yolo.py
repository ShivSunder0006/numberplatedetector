"""
Train YOLOv8 on Kaggle with the 'Large License Plate Dataset'.

Kaggle setup:
- Add the dataset "large-license-plate-dataset" as an input to your notebook.
- Ensure this repo is available (uploaded as a Kaggle notebook or copied into /kaggle/working).
- Run this script in a Kaggle cell:  !python /kaggle/working/license-plate-detector/src/train_yolo.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Paths (Kaggle defaults)
ROOT = Path("/kaggle/working/license-plate-detector")
DATA_YAML = ROOT / "config" / "dataset.yaml"
RUNS_DIR = Path("/kaggle/working/runs")  # so artifacts show up for download
RUN_NAME = "lp_yolov8"

def main():
    print("Using dataset:", DATA_YAML)
    model = YOLO("yolov8n.pt")  # start from a small model; upgrade to s/m/l if you have more GPU
    model.train(
        data=str(DATA_YAML),
        epochs=50,            # adjust for your budget
        imgsz=640,
        batch=16,
        project=str(RUNS_DIR / "detect"),
        name=RUN_NAME,
        pretrained=True,
        device=0,            # use GPU
        exist_ok=True,
        patience=20,
        optimizer="auto",
        lr0=0.01
    )

    # Validate
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        project=str(RUNS_DIR / "detect"),
        name=f"{RUN_NAME}_val",
        device=0
    )
    print("Validation metrics:", metrics)

    # Export best weights path for convenience
    best_path = RUNS_DIR / "detect" / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights: {best_path}")

    # Make a copy in /kaggle/working for easy download via "Output" tab
    out_copy = Path("/kaggle/working") / "best_lp_yolov8.pt"
    if best_path.exists():
        os.system(f"cp {best_path} {out_copy}")
        print(f"Copied best weights to: {out_copy}")
    else:
        print("Warning: best.pt not found. Check run directory.")

if __name__ == "__main__":
    main()
