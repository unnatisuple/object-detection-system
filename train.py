"""
train.py — fine-tune YOLOv8 on your own custom dataset.

Before running this:
  1. Label your images (see README.md for tools/instructions).
  2. Arrange them as described in data/data.yaml and update that file's
     class names + paths to match your dataset.
  3. Run:  python train.py

When training finishes, copy the resulting weights:
  runs/detect/<name>/weights/best.pt  -->  models/best.pt
so the Streamlit app can find them.
"""

from ultralytics import YOLO

# Starting point: a small pretrained YOLOv8 model. We fine-tune it on your
# data rather than training from scratch — much faster and needs far fewer
# images to get good results.
BASE_MODEL = "yolov8n.pt"

DATA_CONFIG = "data/data.yaml"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
RUN_NAME = "custom_model"


def main() -> None:
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
    )

    print(
        f"\nTraining complete.\n"
        f"Copy 'runs/detect/{RUN_NAME}/weights/best.pt' to 'models/best.pt' "
        f"so app.py can load it.\n"
    )


if __name__ == "__main__":
    main()
