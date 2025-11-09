import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Klasifikasi Citra Asli vs Generatif AI", layout="centered")
st.title("🧠 Klasifikasi Citra Asli vs Generatif AI Berbasis CNN")
st.markdown("Aplikasi ini menggunakan tiga model CNN: **ResNet50**, **Xception**, dan **EfficientNetB0**")

# Load model (gunakan cache agar lebih efisien)
@st.cache_resource
def load_models():
    resnet = tf.keras.models.load_model("resnet50_model.h5")
    xception = tf.keras.models.load_model("xception_model.h5")
    efficient = tf.keras.models.load_model("efficientnetb0_model.h5")
    return resnet, xception, efficient

resnet, xception, efficient = load_models()

# Preprocessing image
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Upload file
uploaded = st.file_uploader("📤 Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    img_array = preprocess_image(img)

    if st.button("🔍 Klasifikasi Gambar"):
        st.info("Memproses dengan ketiga model...")

        resnet_pred = resnet.predict(img_array)
        xception_pred = xception.predict(img_array)
        efficient_pred = efficient.predict(img_array)

        classes = ["Asli", "Generatif AI"]

        resnet_class = classes[int(resnet_pred[0][0] > 0.5)]
        xception_class = classes[int(xception_pred[0][0] > 0.5)]
        efficient_class = classes[int(efficient_pred[0][0] > 0.5)]

        avg_pred = (resnet_pred + xception_pred + efficient_pred) / 3
        final_class = classes[int(avg_pred[0][0] > 0.5)]

        st.subheader("📊 Hasil Prediksi Tiap Model:")
        st.write(f"🧩 ResNet50 → **{resnet_class}** ({resnet_pred[0][0]:.2f})")
        st.write(f"🧩 Xception → **{xception_class}** ({xception_pred[0][0]:.2f})")
        st.write(f"🧩 EfficientNetB0 → **{efficient_class}** ({efficient_pred[0][0]:.2f})")

        st.markdown("---")
        st.success(f"🧠 **Prediksi Akhir (Voting 3 Model): {final_class}**")
        st.progress(float(avg_pred[0][0]))
