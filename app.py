import streamlit as st
from PIL import Image
import time

# Настройка страницы
st.set_page_config(page_title="Crossview geolocalization", layout="centered")

def mock_ai_model(image, mode):
    # Имитация работы нейросети
    time.sleep(2) 
    return f"Анализ завершен для режима: {mode}."

# Интерфейс боковой панели
st.sidebar.header("Настройки")
mode = st.sidebar.selectbox(
    "Выберите тип съемки:",
    ("Фото с дрона", "Фото со спутника")
)

# Основной экран
st.title("Поиск местоположения")
st.write(f"Текущий режим: **{mode}**")

uploaded_file = st.file_uploader("Выберите изображение местности...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие и отображение изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_container_width=True)
    
    # Кнопка запуска анализа
    if st.button('Запустить анализ'):
        with st.spinner('Обработка...'):
            # Вызов функции-заглушки
            result = mock_ai_model(image, mode)
            
            # Вывод результата
            st.success('Готово!')
            st.write(result)
            st.info("Здесь появятся результаты, когда модель будет готова.")
else:
    st.info("Пожалуйста, загрузите изображение, чтобы начать.")