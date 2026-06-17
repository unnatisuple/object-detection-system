# Real-Time Custom Object Detector

A small Streamlit app that detects objects live through your webcam using a
YOLOv8 model trained on **your own** images — deployable for free on
Streamlit Community Cloud.

How the webcam works once deployed: the browser's camera feed is sent to the
server over WebRTC (via `streamlit-webrtc`), processed frame-by-frame by your
model, and the annotated video is sent back. This works whether you run it
locally or on the cloud, since it's always the *visitor's* camera, not the
server's.

## Project structure

```
object-detection-app/
├── app.py              # Streamlit app (run this)
├── train.py             # Script to train your custom model
├── requirements.txt      # Python dependencies
├── packages.txt          # System libraries needed by Streamlit Cloud
├── data/data.yaml        # Dataset config — edit with your class names
└── models/best.pt        # Your trained weights go here (after training)
```

## 1. Label your dataset

You need a folder of images with bounding-box labels in YOLO format. Easiest
options:

- **Roboflow** (roboflow.com) — free, browser-based, label and export
  directly in "YOLOv8" format. Recommended if you're new to this.
- **LabelImg** — free desktop tool, export format set to YOLO.

Aim for at least ~50-100 labeled images per class to start; more is better.

Arrange the exported data like this in the project root:

```
datasets/my_dataset/
├── images/train/   images/val/
└── labels/train/   labels/val/
```

Then open `data/data.yaml` and edit the `names:` list to match your actual
classes, in the same order you used while labeling.

## 2. Set up locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train

```bash
python train.py
```

This fine-tunes a small pretrained YOLOv8 model on your dataset (much faster
than training from scratch). When it finishes, copy the result into place:

```bash
cp runs/detect/custom_model/weights/best.pt models/best.pt
```

## 4. Run the app locally

```bash
streamlit run app.py
```

Open the local URL it prints, click **Start**, and allow camera access.

## 5. Push to GitHub

```bash
git init
git add .
git commit -m "Custom object detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Make sure `models/best.pt` is actually committed — it's small (a few MB for
the nano model) and is **not** excluded by `.gitignore`. The raw `datasets/`
folder is excluded on purpose since it can be large and isn't needed once
the model is trained.

## 6. Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **New app**, pick your repo/branch, and set the main file to
   `app.py`.
3. Deploy. Streamlit Cloud automatically installs `requirements.txt` and the
   system packages listed in `packages.txt`.

### If the webcam doesn't connect after deploying

This is usually a NAT/firewall issue with WebRTC's peer connection. Get a
free TURN server (e.g. from
[metered.ca/tools/openrelay](https://www.metered.ca/tools/openrelay) or a
Twilio trial account) and add it to the `iceServers` list in `app.py`:

```python
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:<your-turn-server>"], "username": "...", "credential": "..."},
    ]
})
```

### If the app is slow or runs out of memory

Streamlit Community Cloud's free tier has limited RAM. The nano model
(`yolov8n.pt`) used here is the lightest option. If you trained on a bigger
base model (s/m/l/x) and hit memory issues, retrain starting from
`yolov8n.pt` instead, or export your model to ONNX for lighter inference.

## Notes

- Confidence threshold is adjustable live from the sidebar.
- Class names shown in the bounding boxes come straight from your trained
  model — no need to hardcode them anywhere in the app.
