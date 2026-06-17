"""
Real-Time Custom Object Detection — Streamlit + YOLOv8 + WebRTC

This app streams video straight from the VISITOR'S browser webcam (via WebRTC),
runs each frame through your custom-trained YOLOv8 model on the server, and
sends the annotated frame back. This is what makes "real-time webcam detection"
possible even when the app is deployed on Streamlit Community Cloud, which has
no camera of its own.
"""

import av
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

st.set_page_config(page_title="Custom Object Detector", page_icon="🎯", layout="wide")


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


# ICE servers used to establish the WebRTC connection.
# The public STUN server is enough for local testing and many networks.
# If the stream fails to connect once deployed (common behind strict firewalls/NAT),
# add a free TURN server (e.g. from https://www.metered.ca/tools/openrelay or Twilio)
# as an extra entry in the "iceServers" list below.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.confidence = 0.5
        self.model = load_model(MODEL_PATH)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.confidence, verbose=False)
        annotated = results[0].plot()  # returns a BGR numpy array, ready to send back as-is
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def main() -> None:
    st.title("🎯 Real-Time Custom Object Detection")
    st.caption("Detects objects from your own trained YOLOv8 model, live, in your browser.")

    with st.sidebar:
        st.header("Settings")
        confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        st.markdown("---")
        st.markdown(
            "**Model:** loaded from `models/best.pt`\n\n"
            "Train your own with `train.py` (see README) and drop the resulting "
            "`best.pt` into the `models/` folder."
        )

    st.info("Click **Start** below and allow camera access in your browser.")

    ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.confidence = confidence


if __name__ == "__main__":
    main()
