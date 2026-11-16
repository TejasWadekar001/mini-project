"""Web app:  streamlit run app.py"""
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from infer import Predictor
from report import build_report

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Detection")
st.caption("Upload a leaf photo to get a diagnosis, Grad-CAM explanation and a downloadable disease report.")

runs = sorted(p.parent.name for p in Path("runs").glob("*/best.keras"))
with st.sidebar:
    st.header("Settings")
    if not runs:
        st.error("No trained model found. Train one first (see README).")
        st.stop()
    run = st.selectbox("Trained model", runs, index=runs.index("A5_full") if "A5_full" in runs else 0)
    split = st.text_input("Splits file", "runs/splits.json")
    hier = st.checkbox("Crop-consistent decoding", value=True, help="Only used by models trained with --hier")
    st.markdown("**Tips:** one leaf, good light, leaf filling most of the frame.")


@st.cache_resource(show_spinner="Loading model...")
def load(run_name, split_file):
    return Predictor(Path("runs") / run_name, split_file)


pred = load(run, split)
tab_up, tab_cam = st.tabs(["Upload image", "Use camera"])
with tab_up:
    f = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
with tab_cam:
    c = st.camera_input("Take a photo")
src = f or c
if src is None:
    st.info("Upload or capture a leaf image to begin.")
    st.stop()

with st.spinner("Analysing..."):
    r = pred.analyse(Image.open(src), hier_decode=hier)

st.subheader("Diagnosis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Crop", r["crop"])
m2.metric("Condition", "Healthy" if r["healthy"] else r["disease"])
m3.metric("Confidence", f"{r['confidence'] * 100:.1f}%")
m4.metric("Severity", "None" if r["healthy"] else f"{r['severity_level']} (~{r['severity_pct']:.0f}%)")
if r["uncertain"]:
    st.warning("Low confidence. Retake the photo and confirm with an expert before acting.")
elif r["healthy"]:
    st.success("The leaf looks healthy.")

cols = st.columns(3 if r["attention"] is not None else 2)
cols[0].image(r["image"], caption="Uploaded leaf", use_container_width=True)
cols[1].image(r["gradcam"], caption="Grad-CAM: areas driving the diagnosis", use_container_width=True)
if r["attention"] is not None:
    cols[2].image(r["attention"], caption="Model attention map", use_container_width=True)

st.subheader("Top predictions")
for (crop, dis), p in r["top3"]:
    st.progress(min(max(p, 0.0), 1.0), text=f"{crop}: {dis}  ({p * 100:.1f}%)")

info = r["info"]
st.subheader("Disease information")
st.markdown(f"**Cause:** {info['cause']} ({info['type']})")
st.markdown(f"**Symptoms:** {info['symptoms']}")
a, b = st.columns(2)
a.markdown("**Treatment**\n" + "\n".join(f"- {x}" for x in info["treatment"]))
b.markdown("**Prevention**\n" + "\n".join(f"- {x}" for x in info["prevention"]))

buf = io.BytesIO()
build_report(buf, r)
st.download_button("⬇️ Download PDF disease report", buf.getvalue(),
                   file_name=f"disease_report_{r['label']}.pdf", mime="application/pdf")
st.caption(r["disclaimer"])
