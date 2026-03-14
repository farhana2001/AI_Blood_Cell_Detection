import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ------------------ Page Config ------------------

st.markdown("""
<style>
.stApp {
background-image: url("https://img.freepik.com/premium-photo/blood-cells-with-hemoglobin-abstract-background_144356-950.jpg");
background-size: cover;
background-position: center;
}
</style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="AI Hematology Analyzer",
    layout="centered"
)

st.title("🩸 AI Blood Cell Detection")
st.write("Upload a blood smear image to detect RBC, WBC, and Platelets")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return YOLO("C:\\Users\\VICTUS\\Desktop\\data science\\deep learning projects\\BLOOD_DETECTION\\runs\\detect\\train\\weights\\best.pt")

model = load_model()

# ------------------ Classes ------------------
CLASS_NAMES = {
    0: "Platelets",
    1: "RBC",
    2: "WBC"
}

COLORS = {
    "Platelets": (255,215,0),
    "RBC": (220,53,69),
    "WBC": (30,144,255)
}

# ------------------ Paths ------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSED_IMAGE = os.path.join(OUTPUT_DIR,"processed.jpg")
CHART_IMAGE = os.path.join(OUTPUT_DIR,"chart.png")
PDF_REPORT = os.path.join(OUTPUT_DIR,"blood_report.pdf")

# ------------------ Upload ------------------
uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg","png","jpeg"])

latest_counts = {}

# ------------------ Prediction ------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model(img)[0]

    counts = {name:0 for name in CLASS_NAMES.values()}

    for box in results.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = CLASS_NAMES[cls_id]
        counts[class_name] += 1

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = COLORS[class_name]

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        label = f"{class_name} {conf*100:.1f}%"
        cv2.putText(img,label,(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,color,2)

    cv2.imwrite(PROCESSED_IMAGE,img)
    latest_counts = counts

    st.subheader("Detection Result")
    st.image(img,channels="BGR")

    # ------------------ Counts ------------------
    st.subheader("Cell Counts")

    col1,col2,col3 = st.columns(3)
    col1.metric("RBC",counts["RBC"])
    col2.metric("WBC",counts["WBC"])
    col3.metric("Platelets",counts["Platelets"])

    # ------------------ Chart ------------------
    st.subheader("Cell Distribution")

    fig, ax = plt.subplots()
    ax.bar(counts.keys(),counts.values())
    ax.set_ylabel("Count")
    ax.set_title("Blood Cell Distribution")

    plt.tight_layout()
    plt.savefig(CHART_IMAGE)

    st.pyplot(fig)

# ------------------ PDF Generation ------------------
def generate_pdf():

    doc = SimpleDocTemplate(PDF_REPORT,pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI Hematology Analysis Report</b>",styles["Title"]))
    story.append(Spacer(1,12))

    date_str = datetime.now().strftime("%d %b %Y | %H:%M")
    story.append(Paragraph(f"Generated on: {date_str}",styles["Normal"]))
    story.append(Spacer(1,12))

    story.append(Paragraph("<b>Processed Blood Smear Image</b>",styles["Heading2"]))
    story.append(RLImage(PROCESSED_IMAGE,width=400,height=250))
    story.append(Spacer(1,14))

    table_data=[["Cell Type","Count"]]
    for k,v in latest_counts.items():
        table_data.append([k,str(v)])

    story.append(Paragraph("<b>Detection Summary</b>",styles["Heading2"]))
    story.append(Table(table_data))
    story.append(Spacer(1,14))

    story.append(Paragraph("<b>Cell Distribution Chart</b>",styles["Heading2"]))
    story.append(RLImage(CHART_IMAGE,width=300,height=200))
    story.append(Spacer(1,20))

    story.append(Paragraph(
        "This report is generated using an AI-based system and is intended for research and educational purposes only.",
        styles["Normal"]
    ))

    doc.build(story)

# ------------------ Download Button ------------------
if latest_counts:

    generate_pdf()

    with open(PDF_REPORT,"rb") as f:
        st.download_button(
            label="📄 Download PDF Report",
            data=f,
            file_name="blood_report.pdf",
            mime="application/pdf"
        )