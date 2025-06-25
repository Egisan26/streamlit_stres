import streamlit as st
import numpy as np
import joblib

# ====== CSS Custom Style ======
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6f0;
    }
    * {
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #cc0066;
    }
    </style>
""", unsafe_allow_html=True)

# ====== Load model dan scaler ======
model = joblib.load("Model_stres.pkl")
scaler = joblib.load("scaler_stres.pkl")

# Mapping angka ke label teks
label_map = {
    0: "Rendah",
    1: "Sedang",
    2: "Tinggi"
}

# ====== Judul aplikasi ======
st.title("🧠 Prediksi Tingkat Stres 💆‍♀")
st.write("Masukkan data berikut untuk mengetahui tingkat stres kamu:")

# ====== Input dari pengguna (dengan emoji) ======
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
temperature_f = st.number_input("🌡 Temperature (°F)", min_value=30.0, max_value=130.0, step=0.1)
step_count = st.number_input("👣 Step count", min_value=0, step=1)

# Konversi Fahrenheit ke Celsius
temperature_c = (temperature_f - 32) * 5.0 / 9.0

# ====== Prediksi saat tombol ditekan ======
if st.button("🔍 Prediksi"):
    input_data = np.array([humidity, temperature_f, step_count]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    label = label_map.get(prediction[0], "Tidak Diketahui")

    st.subheader("📊 Hasil Prediksi:")

    # Tampilkan hasil prediksi dengan emoji stres/pusing
    if label == "Rendah":
        st.markdown("😌 Tingkat stres kamu Rendah ")
    elif label == "Sedang":
        st.markdown("😵‍💫 Tingkat stres kamu Sedang ")
    elif label == "Tinggi":
        st.markdown("🥵 Tingkat stres kamu Tinggi ")
    else:
        st.markdown("❓ Hasil tidak diketahui. Coba lagi.")
