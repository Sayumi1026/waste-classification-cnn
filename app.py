
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import hashlib

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F8F5;
    color: #26382B;
}

[data-testid="stHeader"] {
    background-color: #F5F8F5;
}

h1, h2, h3 {
    color: #176B3A !important;
}

p, label, li, span {
    color: #26382B;
}

.hero {
    background: linear-gradient(120deg, #176B3A, #32965B);
    padding: 35px 25px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    color: white !important;
    font-size: 17px;
    margin-bottom: 0;
}

.section-title {
    color: #176B3A;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-card {
    background-color: #E4F4E8;
    border: 2px solid #9BCDA7;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 10px;
}

.result-card h2 {
    color: #176B3A !important;
    font-size: 32px;
    margin: 5px 0;
}

.result-card h3 {
    color: #285D38 !important;
    font-size: 23px;
}

.small-note {
    color: #526B59;
    font-size: 13px;
}

.category-card {
    background-color: white;
    border: 1px solid #D7E5D9;
    border-radius: 12px;
    padding: 17px;
    text-align: center;
    min-height: 125px;
    margin-bottom: 12px;
}

.category-card h4 {
    color: #176B3A !important;
    margin: 7px 0;
}

.category-card p {
    color: #526B59 !important;
    font-size: 13px;
    margin: 0;
}

.stButton > button {
    background-color: #197640;
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #105B30;
    color: white !important;
    border: none;
}

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #76A985;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #26382B !important;
}

[data-testid="stAlert"] {
    color: #26382B !important;
}

hr {
    border-color: #D6E3D8;
}

.footer {
    text-align: center;
    color: #526B59;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>♻️ EcoSort AI</h1>
    <p>
        Smart Waste Classification Using Artificial Intelligence
    </p>
    <p>
        Upload an image, identify the waste category,
        and make smarter waste-sorting decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. WASTE CATEGORY INFORMATION
# =====================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

descriptions = {
    "cardboard": "Boxes, cartons and cardboard packaging.",
    "glass": "Glass bottles, jars and containers.",
    "metal": "Metal cans and other metal objects.",
    "paper": "Newspapers, paper sheets and magazines.",
    "plastic": "Plastic bottles and packaging.",
    "trash": "General waste outside the other categories."
}

category_icons = {
    "cardboard": "📦",
    "glass": "🍾",
    "metal": "🥫",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️"
}


# =====================================================
# 5. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/final_mobilenetv2.keras"
    )


try:

    model = load_model()

except Exception as error:

    st.error(
        "The trained model could not be loaded. "
        "Please check the model file inside the models folder."
    )

    st.exception(error)
    st.stop()


# =====================================================
# 6. SESSION STATE
# =====================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "image_signature" not in st.session_state:
    st.session_state.image_signature = None


# =====================================================
# 7. USER INSTRUCTIONS
# =====================================================

with st.expander("ℹ️ How to use EcoSort AI"):

    st.markdown("""
    **Follow these simple steps:**

    1. Select an image of a waste item.
    2. Preview the uploaded image.
    3. Click the **Classify My Waste** button.
    4. View the predicted category and confidence.
    5. Check the probability breakdown.

    **Supported image formats:** JPG, JPEG and PNG.

    **Supported categories:** Cardboard, Glass, Metal,
    Paper, Plastic and Trash.
    """)


st.divider()


# =====================================================
# 8. IMAGE UPLOAD SECTION
# =====================================================

st.markdown(
    '<div class="section-title">📸 Step 1: Upload Your Waste Image</div>',
    unsafe_allow_html=True
)

st.caption(
    "Choose a clear image of the waste item you want to identify."
)

uploaded_file = st.file_uploader(
    "Choose an image from your device",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG or PNG image.",
    key="waste_image"
)


# =====================================================
# 9. RESET RESULT WHEN IMAGE CHANGES
# =====================================================

if uploaded_file is not None:

    current_signature = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if current_signature != st.session_state.image_signature:

        st.session_state.prediction_result = None

        st.session_state.image_signature = current_signature


# =====================================================
# 10. IMAGE PREVIEW AND CLASSIFICATION
# =====================================================

if uploaded_file is None:

    st.info(
        "👆 Start by uploading an image of cardboard, glass, "
        "metal, paper, plastic or general waste."
    )

else:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="section-title">👀 Step 2: Preview Your Image</div>',
            unsafe_allow_html=True
        )

        image_col, action_col = st.columns(
            [1.2, 1],
            gap="large"
        )

        with image_col:

            st.image(
                image,
                caption="Your selected waste image",
                use_container_width=True
            )

        with action_col:

            st.markdown(
                '<div class="section-title">🤖 Ready to Identify?</div>',
                unsafe_allow_html=True
            )

            st.write(
                "Our trained AI model will analyze the image "
                "and predict its most likely waste category."
            )

            st.write("")

            classify_button = st.button(
                "♻️ Classify My Waste",
                use_container_width=True
            )

            if st.session_state.prediction_result is not None:

                if st.button(
                    "🔄 Clear Prediction",
                    use_container_width=True
                ):

                    st.session_state.prediction_result = None
                    st.rerun()

            st.caption(
                "Tip: Use a clear image where the waste item "
                "is visible."
            )

        # =================================================
        # 11. RUN PREDICTION
        # =================================================

        if classify_button:

            with st.spinner(
                "🔍 EcoSort AI is analyzing your image..."
            ):

                # Convert image into array
                image_array = np.array(image)

                # Convert to TensorFlow tensor
                image_tensor = tf.convert_to_tensor(
                    image_array,
                    dtype=tf.float32
                )

                # Resize while preserving aspect ratio
                image_tensor = tf.image.resize_with_pad(
                    image_tensor,
                    224,
                    224
                )

                # Add batch dimension
                image_tensor = tf.expand_dims(
                    image_tensor,
                    axis=0
                )

                # Predict
                predictions = model.predict(
                    image_tensor,
                    verbose=0
                )[0]

                # Get predicted category
                predicted_index = int(np.argmax(predictions))

                predicted_class = class_names[predicted_index]

                confidence = float(
                    predictions[predicted_index] * 100
                )

                # Store results
                st.session_state.prediction_result = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "probabilities": predictions.tolist()
                }

        # =================================================
        # 12. DISPLAY PREDICTION RESULT
        # =================================================

        if st.session_state.prediction_result is not None:

            result = st.session_state.prediction_result

            predicted_class = result["class"]

            confidence = result["confidence"]

            predictions = np.array(
                result["probabilities"]
            )

            st.divider()

            st.markdown(
                '<div class="section-title">✅ Step 3: Your Classification Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size:48px;">
                        {category_icons[predicted_class]}
                    </div>
                    <h2>{predicted_class.capitalize()}</h2>
                    <h3>{confidence:.2f}% Confidence</h3>
                    <p style="color:#285D38;">
                        {descriptions[predicted_class]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            # Confidence explanation
            st.caption(
                "Confidence represents the model's predicted "
                "probability for this category. It does not "
                "guarantee that the prediction is correct."
            )

            # =================================================
            # 13. PROBABILITY BREAKDOWN
            # =================================================

            st.markdown(
                '<div class="section-title">📊 Prediction Breakdown</div>',
                unsafe_allow_html=True
            )

            st.write(
                "See how the model evaluated each waste category."
            )

            probability_df = pd.DataFrame({
                "Category": [
                    category.capitalize()
                    for category in class_names
                ],
                "Probability": predictions * 100
            })

            # Horizontal bar chart
            fig, ax = plt.subplots(figsize=(9, 4.5))

            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            bar_colors = [
                "#197640" if category == predicted_class
                else "#A9D5B4"
                for category in class_names
            ]

            bars = ax.barh(
                probability_df["Category"],
                probability_df["Probability"],
                color=bar_colors,
                edgecolor="#FFFFFF",
                height=0.6
            )

            ax.set_xlabel(
                "Probability (%)",
                fontsize=11,
                color="#26382B"
            )

            ax.set_title(
                "Probability by Waste Category",
                fontsize=14,
                color="#176B3A",
                fontweight="bold"
            )

            ax.set_xlim(0, 110)

            ax.tick_params(
                axis="both",
                colors="#26382B"
            )

            ax.grid(
                axis="x",
                linestyle="--",
                alpha=0.25
            )

            ax.set_axisbelow(True)

            for bar in bars:

                width = bar.get_width()

                ax.text(
                    width + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}%",
                    va="center",
                    fontsize=9,
                    color="#26382B"
                )

            for spine in ax.spines.values():
                spine.set_visible(False)

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # =================================================
            # 14. DETAILED PROBABILITY RESULTS
            # =================================================

            with st.expander("View detailed prediction values"):

                details_df = pd.DataFrame({
                    "Waste Category": [
                        category.capitalize()
                        for category in class_names
                    ],
                    "Probability (%)": [
                        f"{value * 100:.2f}%"
                        for value in predictions
                    ]
                })

                st.dataframe(
                    details_df,
                    use_container_width=True,
                    hide_index=True
                )


    except Exception as error:

        st.error(
            "Something went wrong while processing the image. "
            "Please try another image."
        )

        st.exception(error)


# =====================================================
# 15. SUPPORTED WASTE CATEGORIES
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">🌱 What Can EcoSort AI Identify?</div>',
    unsafe_allow_html=True
)

st.write(
    "The model can classify images into the following six categories:"
)

category_columns = st.columns(3)

for index, category in enumerate(class_names):

    with category_columns[index % 3]:

        st.markdown(
            f"""
            <div class="category-card">
                <div style="font-size:30px;">
                    {category_icons[category]}
                </div>
                <h4>{category.capitalize()}</h4>
                <p>{descriptions[category]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    <strong>♻️ EcoSort AI</strong>
    <br>
    Automated Waste Classification Using Deep Learning
    <br><br>
    Developed using CNN, MobileNetV2 and Streamlit
    <br>
    © 2026 EcoSort AI
</div>
""", unsafe_allow_html=True)
