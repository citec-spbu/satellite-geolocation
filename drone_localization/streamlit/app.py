import os
import base64
import io

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🛰️ Satellite Geolocation")
uploaded = st.file_uploader("Загрузи фото с дрона", type=["jpg", "png"])

if uploaded:
    st.image(uploaded, caption="Фото с дрона", use_column_width=True)

    if st.button("Найти координаты"):
        with st.spinner("Ищу спутниковый снимок и уточняю позицию..."):
            # Конвертация в base64
            img_bytes = uploaded.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Запрос к API
            res = requests.post(f"{API_URL}/api/localize", json={"drone_image": img_b64})

            if res.status_code == 200:
                data = res.json()
                # Показываем найденный спутник
                sat_bytes = base64.b64decode(data["satellite_image"])
                st.image(Image.open(io.BytesIO(sat_bytes)), caption="Найденный спутник")

                st.success(
                    f"Координаты: {data['coordinates']['lat']:.5f}, {data['coordinates']['lon']:.5f}"
                )
                st.metric("Confidence", f"{data['confidence']:.2f}")
            else:
                st.error(f"Ошибка API: {res.json()['detail']}")
