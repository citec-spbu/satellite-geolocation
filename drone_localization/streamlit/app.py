import os
import base64
import io
import json

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="🛰️ Satellite Geolocation", page_icon="🛰️", layout="wide")

st.title("🛰️ Satellite Geolocation Platform")
st.markdown("Платформа для геолокации дронов и управления спутниковыми изображениями")

# Боковое меню с навигацией по всем страницам
st.sidebar.header("Меню")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["🛰️ Геолокация дрона", "📤 Загрузить изображение", "🔍 Поиск похожих", "📊 Просмотр галереи", "⚙️ Управление"]
)

# Helper функции
def encode_image(file) -> str:
    """Кодирует файл в base64 строку"""
    return base64.b64encode(file.read()).decode("utf-8")


def decode_image(base64_str: str) -> Image.Image:
    """Декодирует base64 строку в PIL Image"""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))


# ==================== ГЕОЛОКАЦИЯ ДРОНА ====================
if page == "🛰️ Геолокация дрона":
    st.header("🛰️ Геолокация дрона по спутниковым снимкам")
    st.markdown("Загрузите фото с дрона для определения координат")

    uploaded = st.file_uploader("Загрузи фото с дрона", type=["jpg", "png"])

    if uploaded:
        st.image(uploaded, caption="Фото с дрона", use_column_width=True)

        if st.button("Найти координаты", type="primary"):
            with st.spinner("Ищу спутниковый снимок и уточняю позицию..."):
                # Конвертация в base64
                img_bytes = uploaded.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                # Запрос к API
                res = requests.post(
                    f"{API_URL}/api/localize", json={"drone_image": img_b64}
                )

                if res.status_code == 200:
                    data = res.json()
                    # Показываем найденный спутник
                    sat_bytes = base64.b64decode(data["satellite_image"])
                    st.image(Image.open(io.BytesIO(sat_bytes)), caption="Найденный спутник")

                    st.success(
                        f"Координаты: {data['coordinates']['lat']:.5f}, {data['coordinates']['lon']:.5f}"
                    )
                    st.metric("Confidence", f"{data['confidence']:.2f}")

                    # Показываем ссылку на Google Maps
                    lat, lon = data['coordinates']['lat'], data['coordinates']['lon']
                    st.markdown(f"[🗺️ Открыть в Google Maps](https://www.google.com/maps?q={lat},{lon})")
                else:
                    st.error(f"Ошибка API: {res.json().get('detail', 'Неизвестная ошибка')}")


# ==================== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ====================
if page == "📤 Загрузить изображение":
    st.header("📤 Загрузить новое изображение")
    st.markdown("Загрузите спутниковое изображение для добавления в галерею")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Выберите файл изображения",
            type=["jpg", "jpeg", "png"],
            help="Поддерживаются форматы: JPEG, PNG"
        )

    with col2:
        st.markdown("### Метаданные (опционально)")
        metadata_filename = st.text_input("Имя файла", "")
        metadata_location = st.text_input("Локация", "")

        st.markdown("#### Координаты прямоугольника (верхний левый и нижний правый углы)")
        col_tl_E, col_tl_N = st.columns(2)
        with col_tl_E:
            tl_E = st.number_input("Долгота верхнего левого угла (tl_E)", min_value=-180.0, max_value=180.0, value=None, step=0.000001)
        with col_tl_N:
            tl_N = st.number_input("Широта верхнего левого угла (tl_N)", min_value=-90.0, max_value=90.0, value=None, step=0.000001)

        col_br_E, col_br_N = st.columns(2)
        with col_br_E:
            br_E = st.number_input("Долгота нижнего правого угла (br_E)", min_value=-180.0, max_value=180.0, value=None, step=0.000001)
        with col_br_N:
            br_N = st.number_input("Широта нижнего правого угла (br_N)", min_value=-90.0, max_value=90.0, value=None, step=0.000001)

        metadata_custom = st.text_area("Дополнительные метаданные (JSON)", "{}")

    if uploaded_file:
        # Показываем превью
        image = Image.open(uploaded_file)
        st.image(image, caption="Превью загружаемого изображения", use_column_width=True)

        # Подготовка метаданных
        metadata = {}
        if metadata_filename:
            metadata["filename"] = metadata_filename
        if metadata_location:
            metadata["location"] = metadata_location
        if metadata_custom and metadata_custom.strip() != "{}":
            try:
                custom_meta = json.loads(metadata_custom)
                metadata.update(custom_meta)
            except json.JSONDecodeError:
                st.error("Некорректный JSON в дополнительных метаданных")
                metadata = {}

        # Добавляем координаты если указаны
        coordinates = None
        if all(v is not None for v in [tl_E, tl_N, br_E, br_N]):
            coordinates = {"tl_E": tl_E, "tl_N": tl_N, "br_E": br_E, "br_N": br_N}

        if st.button("Загрузить в галерею", type="primary"):
            with st.spinner("Загрузка изображения..."):
                try:
                    # Кодируем изображение
                    uploaded_file.seek(0)
                    img_b64 = encode_image(uploaded_file)

                    # Отправляем запрос с координатами в правильном формате
                    payload = {"image": img_b64}
                    if coordinates:
                        # Формат coordinates должен быть объектом с tl_E, tl_N, br_E, br_N
                        payload["coordinates"] = {
                            "tl_E": coordinates["tl_E"],
                            "tl_N": coordinates["tl_N"],
                            "br_E": coordinates["br_E"],
                            "br_N": coordinates["br_N"]
                        }
                    if metadata:
                        payload["metadata"] = metadata

                    response = requests.post(
                        f"{API_URL}/api/gallery/upload",
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Изображение успешно загружено!")
                        st.info(f"**Image ID:** `{result['image_id']}`")

                        # Показываем загруженное изображение
                        get_response = requests.get(f"{API_URL}/api/gallery/image/{result['image_id']}")
                        if get_response.status_code == 200:
                            img_data = get_response.json()
                            loaded_img = decode_image(img_data["image"])
                            st.image(loaded_img, caption=f"Загруженное изображение (ID: {result['image_id']})", use_column_width=True)

                            # Показываем координаты если есть
                            if img_data.get("coordinates"):
                                coords = img_data["coordinates"]
                                st.success(f"📍 Координаты прямоугольника:")
                                st.info(f"**Верхний левый угол:** {coords['tl_N']:.6f}, {coords['tl_E']:.6f}\n\n**Нижний правый угол:** {coords['br_N']:.6f}, {coords['br_E']:.6f}")
                                st.markdown(f"[Открыть в Google Maps](https://www.google.com/maps?q={coords['tl_N']},{coords['tl_E']})")

                            if img_data.get("metadata"):
                                st.json(img_data["metadata"])
                    else:
                        st.error(f"❌ Ошибка загрузки: {response.json().get('detail', 'Неизвестная ошибка')}")

                except Exception as e:
                    st.error(f"❌ Произошла ошибка: {str(e)}")


# ==================== ПОИСК ПОХОЖИХ ====================
elif page == "🔍 Поиск похожих":
    st.header("🔍 Поиск похожих изображений")
    st.markdown("Загрузите изображение для поиска похожих в галерее")

    col1, col2 = st.columns(2)

    with col1:
        search_file = st.file_uploader(
            "Изображение для поиска",
            type=["jpg", "jpeg", "png"],
            key="search_upload"
        )

    with col2:
        top_k = st.slider("Количество результатов", min_value=1, max_value=20, value=5)

    if search_file:
        search_image = Image.open(search_file)
        st.image(search_image, caption="Изображение для поиска", use_column_width=True)

        if st.button("Найти похожие", type="primary"):
            with st.spinner("Поиск похожих изображений..."):
                try:
                    search_file.seek(0)
                    img_b64 = encode_image(search_file)

                    response = requests.post(
                        f"{API_URL}/api/gallery/search",
                        json={"image": img_b64, "top_k": top_k}
                    )

                    if response.status_code == 200:
                        results = response.json()["results"]

                        if not results:
                            st.warning("Похожие изображения не найдены")
                        else:
                            st.success(f"Найдено {len(results)} похожих изображений")

                            # Показываем результаты
                            for i, result in enumerate(results):
                                with st.expander(f"🖼️ Результат #{i+1} (Score: {result['score']:.4f})"):
                                    result_img = decode_image(result["image"])
                                    st.image(result_img, use_column_width=True)
                                    st.code(f"Image ID: {result['image_id']}")

                                    # Показываем координаты если есть
                                    if result.get("coordinates"):
                                        coords = result["coordinates"]
                                        st.success(f"📍 Координаты прямоугольника:")
                                        st.info(f"**Верхний левый угол:** {coords['tl_N']:.6f}, {coords['tl_E']:.6f}\n\n**Нижний правый угол:** {coords['br_N']:.6f}, {coords['br_E']:.6f}")
                                        st.markdown(f"[Открыть в Google Maps](https://www.google.com/maps?q={coords['tl_N']},{coords['tl_E']})")

                                    if result.get("metadata"):
                                        st.json(result["metadata"])
                    else:
                        st.error(f"❌ Ошибка поиска: {response.json().get('detail', 'Неизвестная ошибка')}")

                except Exception as e:
                    st.error(f"❌ Произошла ошибка: {str(e)}")


# ==================== ПРОСМОТР ГАЛЕРЕИ ====================
elif page == "📊 Просмотр галереи":
    st.header("📊 Просмотр всех изображений в галерее")

    # Получаем количество изображений
    try:
        count_response = requests.get(f"{API_URL}/api/gallery/count")
        if count_response.status_code == 200:
            total_count = count_response.json()["count"]
            st.metric("Всего изображений в галерее", total_count)
        else:
            st.error("Не удалось получить количество изображений")
            total_count = 0
    except Exception as e:
        st.error(f"Ошибка подключения к API: {str(e)}")
        total_count = 0

    if total_count > 0:
        # Запрашиваем health check
        try:
            health_response = requests.get(f"{API_URL}/api/gallery/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                if health_data["healthy"]:
                    st.success("✅ Хранилище работает нормально")
                else:
                    st.warning(f"⚠️ Проблемы с хранилищем: {health_data.get('details', '')}")
        except:
            pass

        st.divider()

        # Поле для ввода ID конкретного изображения
        st.subheader("Просмотр по ID")
        col1, col2 = st.columns([3, 1])
        with col1:
            image_id_input = st.text_input("Введите Image ID", placeholder="Например: abc123...")
        with col2:
            view_btn = st.button("Просмотреть", type="secondary")

        if view_btn and image_id_input:
            with st.spinner("Загрузка изображения..."):
                try:
                    response = requests.get(f"{API_URL}/api/gallery/image/{image_id_input}")
                    if response.status_code == 200:
                        img_data = response.json()
                        img = decode_image(img_data["image"])
                        st.image(img, caption=f"Image ID: {image_id_input}", use_column_width=True)
                        if img_data.get("metadata"):
                            st.json(img_data["metadata"])
                    else:
                        st.error(f"Изображение не найдено: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

        st.divider()

        # Примечание
        st.info("""
        💡 **Подсказка:** Для просмотра всех изображений используйте эндпоинт `/api/gallery/count`
        для получения количества, а затем запрашивайте каждое изображение по ID.

        В будущей версии здесь будет реализована пагинация для просмотра всех изображений.
        """)


# ==================== УПРАВЛЕНИЕ ====================
elif page == "⚙️ Управление":
    st.header("⚙️ Управление галереей")
    st.warning("⚠️ Будьте осторожны! Некоторые действия необратимы.")

    # Статистика
    try:
        count_response = requests.get(f"{API_URL}/api/gallery/count")
        if count_response.status_code == 200:
            count = count_response.json()["count"]
            st.metric("Текущее количество изображений", count)
    except:
        pass

    st.divider()

    # Удаление по ID
    st.subheader("🗑️ Удалить изображение по ID")
    col1, col2 = st.columns([3, 1])
    with col1:
        delete_id = st.text_input("Image ID для удаления", key="delete_input")
    with col2:
        delete_btn = st.button("Удалить", type="secondary", key="delete_btn")

    if delete_btn and delete_id:
        if st.checkbox("Вы уверены? Это действие нельзя отменить.", key="confirm_delete"):
            with st.spinner("Удаление..."):
                try:
                    response = requests.delete(f"{API_URL}/api/gallery/image/{delete_id}")
                    if response.status_code == 200:
                        st.success(f"✅ Изображение {delete_id} успешно удалено")
                    else:
                        st.error(f"Ошибка: {response.json().get('detail', 'Неизвестная ошибка')}")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
        else:
            st.warning("Подтвердите удаление")

    st.divider()

    st.subheader("☢️ Полная очистка галереи")
    st.error("⚠️ Внимание! Это удалит ВСЕ изображения из галереи без возможности восстановления!")

    # Инициализация состояния в сессии
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False

    # Кнопка запускает процесс подтверждения
    if st.button("☢️ ОЧИСТИТЬ ВСЮ ГАЛЕРЕЮ", type="primary", key="clear_all_btn"):
        # Устанавливаем флаг, что пользователь хочет очистить
        st.session_state.confirm_clear = True
        st.rerun()  # Перезагружаем, чтобы показать чекбокс

    # Если флаг установлен, показываем чекбокс и логику удаления
    if st.session_state.confirm_clear:
        st.warning("Подтвердите действие ниже:")
        confirm_checkbox = st.checkbox("Я полностью осознаю последствия. Все изображения будут удалены навсегда.", key="confirm_checkbox_key")

        if confirm_checkbox:
            with st.spinner("Очистка галереи..."):
                try:
                    response = requests.delete(f"{API_URL}/api/gallery/clear")
                    if response.status_code == 200:
                        st.success("✅ Галерея полностью очищена")
                        # Сбрасываем флаг и перезагружаем
                        st.session_state.confirm_clear = False
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {response.json().get('detail', 'Неизвестная ошибка')}")
                        st.session_state.confirm_clear = False # Сброс при ошибке тоже желателен
                except Exception as e:
                    st.error(f"Ошибка соединения: {str(e)}")
                    st.session_state.confirm_clear = False

                # Кнопка отмены, если передумали
                if st.button("Отмена", key="cancel_clear"):
                    st.session_state.confirm_clear = False
                    st.rerun()
        else:
            st.info("Нажмите на чекбокс выше для подтверждения.")

    st.divider()

    # Health check
    st.subheader("🏥 Проверка состояния")
    if st.button("Проверить состояние хранилища", key="health_check"):
        with st.spinner("Проверка..."):
            try:
                response = requests.get(f"{API_URL}/api/gallery/health")
                if response.status_code == 200:
                    data = response.json()
                    if data["healthy"]:
                        st.success("✅ Хранилище работает нормально")
                        st.json(data)
                    else:
                        st.error(f"❌ Проблемы с хранилищем: {data.get('details', 'Неизвестная ошибка')}")
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка подключения: {str(e)}")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Gallery Service UI | Powered by Streamlit & FastAPI</small>
</div>
""", unsafe_allow_html=True)