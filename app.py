import streamlit.components.v1 as components
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #12365b, #061326 65%);
    color: white;
}

h1, h2, h3 {
    color: #42e8d0 !important;
}

p, label, span {
    color: white;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #102d4c, #071b35);
    border: 1px solid #13b9d9;
    border-radius: 15px;
    padding: 15px;
}

.stButton button {
    background: linear-gradient(90deg, #087cf0, #792cf5);
    color: white;
    border: 1px solid #40dfff;
    border-radius: 12px;
    font-weight: bold;
    width: 100%;
    padding: 12px;
}

.stButton button:hover {
    border: 1px solid #55ffdf;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #102b49;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #126a9c;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL AND CLASSES
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "DermaScan_AI_Final.keras"
    )

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as file:
        return json.load(file)

disease_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

try:
    model = load_model()
    class_names = load_classes()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div style="padding:10px 0 20px 0;">
    <h1 style="font-size:48px; margin-bottom:0;">
        🔬 DermaScan <span style="color:#40e8d0;">AI</span>
    </h1>
    <h3 style="color:#b9eaff !important;">
        AI-Based Skin Image Classification
    </h3>
    <p style="color:#68d9f5;">
        Smart Analysis • Better Awareness • Healthier You
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MEDICAL DISCLAIMER
# ==========================================

st.warning("""
**Educational research prototype only.**
This application cannot diagnose skin conditions or determine
whether a lesion is cancerous. Consult a qualified dermatologist
for medical evaluation.
""")

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🔬 DermaScan AI")

    st.markdown("### Project Information")
    st.write("**Model:** MobileNetV2")
    st.write("**Input:** 224 × 224")
    st.write("**Classes:** 7")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("""
    🏠 Home

    📚 About Project

    🔬 Skin Classification

    🛡️ Precautions

    ⚠️ Disclaimer
    """)

    st.divider()

    st.info(
        "This is an educational AI research project. "
        "Model outputs are not medical diagnoses."
    )

# ==========================================



# ==========================================
# DETAILED 3D SKIN CROSS-SECTION
# ==========================================

skin_visual = """
<html>
<body style="margin:0;background:transparent;">
<svg viewBox="0 0 900 390" width="100%"
     xmlns="http://www.w3.org/2000/svg">

<defs>
  <linearGradient id="epi" x2="0" y2="1">
    <stop stop-color="#ffd6ae"/>
    <stop offset=".45" stop-color="#f5a18e"/>
    <stop offset="1" stop-color="#d65e78"/>
  </linearGradient>

  <linearGradient id="dermis" x2="0" y2="1">
    <stop stop-color="#f28c9c"/>
    <stop offset=".5" stop-color="#d96c91"/>
    <stop offset="1" stop-color="#a64d83"/>
  </linearGradient>

  <linearGradient id="fat" x2="0" y2="1">
    <stop stop-color="#ffcf73"/>
    <stop offset="1" stop-color="#dc8c36"/>
  </linearGradient>

  <radialGradient id="hair">
    <stop stop-color="#b47a49"/>
    <stop offset=".7" stop-color="#432419"/>
    <stop offset="1" stop-color="#160c09"/>
  </radialGradient>

  <radialGradient id="cell">
    <stop stop-color="#e8ffff"/>
    <stop offset=".4" stop-color="#46caff"/>
    <stop offset="1" stop-color="#2440a0"/>
  </radialGradient>

  <filter id="shadow">
    <feGaussianBlur stdDeviation="8"/>
  </filter>
</defs>

<!-- Background -->
<rect width="900" height="390" rx="20" fill="#071b34"/>

<text x="28" y="35" fill="#52eaff"
      font-size="23" font-weight="bold">
  3D SKIN ANATOMY
</text>

<!-- Epidermis -->
<path d="M30 125 Q100 95 170 120 T310 120
         T450 120 T590 120 T730 120 T870 120
         L870 165 L30 165 Z"
      fill="url(#epi)" stroke="#ffcfaa" stroke-width="4"/>

<!-- Epidermal cells -->
<g fill="#ffcfbd" opacity=".7">
  <ellipse cx="90" cy="135" rx="23" ry="9"/>
  <ellipse cx="160" cy="145" rx="25" ry="8"/>
  <ellipse cx="240" cy="130" rx="23" ry="9"/>
  <ellipse cx="330" cy="143" rx="25" ry="9"/>
  <ellipse cx="420" cy="132" rx="24" ry="9"/>
  <ellipse cx="510" cy="145" rx="25" ry="9"/>
  <ellipse cx="600" cy="132" rx="23" ry="9"/>
  <ellipse cx="700" cy="145" rx="25" ry="9"/>
  <ellipse cx="800" cy="133" rx="24" ry="9"/>
</g>

<!-- Dermis -->
<path d="M30 165 L870 165 L870 280 L30 280 Z"
      fill="url(#dermis)"/>

<!-- Dermis collagen fibers -->
<g fill="none" stroke-linecap="round" opacity=".65">
  <path d="M35 190 Q180 220 300 180 T570 205 T865 185"
        stroke="#ffbdc5" stroke-width="7"/>
  <path d="M35 230 Q180 190 300 245 T570 220 T865 250"
        stroke="#a94681" stroke-width="8"/>
  <path d="M35 260 Q180 235 300 265 T570 250 T865 265"
        stroke="#ffc0bd" stroke-width="6"/>
</g>

<!-- Blood vessels -->
<g fill="none" stroke-linecap="round">
  <path d="M70 260 Q160 205 240 260 T410 255"
        stroke="#f04475" stroke-width="7"/>
  <path d="M410 255 Q500 205 570 260 T800 250"
        stroke="#e84475" stroke-width="7"/>
  <path d="M240 260 Q260 230 285 230"
        stroke="#ff9fbc" stroke-width="4"/>
  <path d="M570 260 Q590 230 620 230"
        stroke="#ff9fbc" stroke-width="4"/>
</g>

<!-- Subcutaneous fat -->
<path d="M30 280 L870 280 L870 370 L30 370 Z"
      fill="url(#fat)"/>

<g fill="#ffe08c" stroke="#d8893b" stroke-width="3">
  <circle cx="85" cy="315" r="26"/>
  <circle cx="145" cy="345" r="29"/>
  <circle cx="220" cy="315" r="27"/>
  <circle cx="300" cy="345" r="30"/>
  <circle cx="385" cy="315" r="27"/>
  <circle cx="470" cy="345" r="30"/>
  <circle cx="555" cy="315" r="27"/>
  <circle cx="640" cy="345" r="30"/>
  <circle cx="730" cy="315" r="27"/>
  <circle cx="815" cy="345" r="29"/>
</g>

<!-- Hair shaft -->
<path d="M440 190 Q435 115 470 70 Q482 48 475 42"
      stroke="#28130d" stroke-width="15"
      fill="none" stroke-linecap="round"/>

<path d="M440 190 Q435 115 470 70"
      stroke="#9d6340" stroke-width="4"
      fill="none"/>

<!-- Hair follicle -->
<path d="M420 145 Q390 220 425 275
         Q455 315 485 275 Q520 220 480 145"
      fill="#b34c64" stroke="#ffc19c" stroke-width="5"/>

<ellipse cx="452" cy="272" rx="25" ry="34"
         fill="url(#hair)"/>

<!-- Sebaceous gland -->
<g fill="#ffd36d" stroke="#ed9950" stroke-width="3">
  <circle cx="500" cy="180" r="19"/>
  <circle cx="520" cy="164" r="15"/>
  <circle cx="526" cy="190" r="13"/>
</g>

<!-- Microscopic particles -->
<g>
  <circle cx="115" cy="75" r="19" fill="url(#cell)"/>
  <circle cx="115" cy="75" r="8" fill="#d8ffff"/>
  <circle cx="740" cy="80" r="23" fill="url(#cell)"/>
  <circle cx="740" cy="80" r="9" fill="#d8ffff"/>
  <circle cx="810" cy="195" r="14" fill="url(#cell)"/>
  <circle cx="190" cy="90" r="10" fill="#c75ee9"/>
  <circle cx="670" cy="185" r="12" fill="#c75ee9"/>
</g>

<!-- Layer labels -->
<g font-size="17" font-weight="bold" fill="#ffffff">
  <text x="35" y="112">EPIDERMIS</text>
  <text x="35" y="215">DERMIS</text>
  <text x="35" y="355">SUBCUTANEOUS LAYER</text>
</g>

<text x="540" y="110" fill="#b9f5ff" font-size="16">
  Hair follicle
</text>
<path d="M535 115 L480 150" stroke="#b9f5ff"
      stroke-width="2" fill="none"/>

<text x="555" y="205" fill="#ffffff" font-size="15">
  Sebaceous gland
</text>
<path d="M550 200 L520 185" stroke="#ffffff"
      stroke-width="2" fill="none"/>

</svg>
</body>
</html>
"""

components.html(skin_visual, height=400, scrolling=False)


# ==========================================
# IMAGE UPLOAD AND CAMERA
# ==========================================

st.header("📸 Upload or Capture Skin Image")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    camera_file = st.camera_input(
        "Capture Image Using Camera"
    )

image_file = (
    uploaded_file
    if uploaded_file is not None
    else camera_file
)

# ==========================================
# IMAGE PREVIEW
# ==========================================

if image_file is not None:

    try:
        image = Image.open(image_file).convert("RGB")

        st.divider()

        left, right = st.columns([1, 1])

        with left:
            st.subheader("🖼️ Selected Image")
            st.image(
                image,
                caption="Uploaded Skin Image",
                use_container_width=True
            )

        with right:
            st.subheader("🔍 Image Analysis")

            analyze = st.button(
                "🚀 Analyze Image",
                type="primary",
                use_container_width=True
            )

        # ==================================
        # MODEL PREDICTION
        # ==================================

        if analyze:

            with st.spinner("Analyzing image..."):

                resized = image.resize((224, 224))

                array = np.array(
                    resized,
                    dtype=np.float32
                )

                array = np.expand_dims(array, axis=0)
                array = preprocess_input(array)

                predictions = model.predict(
                    array,
                    verbose=0
                )[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "Model output and class names do not match."
                    )
                    st.stop()

                results = sorted(
                    zip(class_names, predictions),
                    key=lambda item: float(item[1]),
                    reverse=True
                )

            st.success("Analysis completed!")

            # ==================================
            # HIGHEST MODEL SCORE
            # ==================================

            best_class, best_score = results[0]

            st.markdown("## 🏆 Highest Model Score")

            st.metric(
                disease_names.get(best_class, best_class),
                f"{float(best_score) * 100:.2f}%"
            )

            st.caption(
                "This score is the model's output, not the "
                "probability of a medical diagnosis."
            )

            # ==================================
            # TOP 3 PREDICTIONS
            # ==================================

            st.divider()
            st.subheader("🥇 Top 3 Predictions")

            for index, (name, score) in enumerate(
                results[:3],
                start=1
            ):

                percentage = float(score) * 100

                st.write(
                    f"**{index}. {disease_names.get(name, name)}**"
                )

                st.progress(float(score))

                st.caption(
                    f"Model score: {percentage:.2f}%"
                )

            # ==================================
            # ALL CLASS SCORES
            # ==================================

            st.divider()
            st.subheader("📊 All Class Scores")

            for name, score in results:

                percentage = float(score) * 100

                st.write(
                    f"**{disease_names.get(name, name)}** "
                    f"— {percentage:.2f}%"
                )

                st.progress(float(score))

            # ==================================
            # FINAL DISCLAIMER
            # ==================================

            st.divider()

            st.info("""
            **Important Medical Notice**

            This AI project is designed for educational research.
            It cannot confirm or rule out any skin disease or cancer.
            Do not make treatment decisions based on these scores.
            Consult a qualified dermatologist for medical evaluation.
            """)

    except Exception as error:
        st.error(f"Image processing error: {error}")

else:
    st.info(
        "👆 Upload a skin image or capture one using your camera."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <h4 style="color:#44e8d0;">
        🔬 DermaScan AI
    </h4>
    <p style="color:#9ccfe8;">
        Educational AI Research Project
    </p>
    <p style="color:#48dfff;">
        Healthy Skin • Better Awareness
    </p>
</div>
""", unsafe_allow_html=True)
