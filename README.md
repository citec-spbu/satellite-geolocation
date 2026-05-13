# drone-localization

## 📖 Описание проекта

**Drone Localization** — это сервис для определения географических координат аэрофотоснимков (с дронов, БПЛА или высотных камер) путём их сопоставления со спутниковой картой.
Проект решает задачу cross-view geo-localization: на вход подаётся изображение «с воздуха», а на выходе сервис возвращает точные GPS-координаты (широта/долгота), оценку уверенности модели и соответствующий спутниковый тайл для визуальной верификации.

## Пайплайн обработки

Система построена на двухэтапной системе **поиск + уточнение** (retrieval + refinement).
![alt text](image_2026-05-05_18-05-25.png)

## 🧠 Используемые модели

### Retrieval

### Refinemen

## 🛠️ Технологический стек

|Компонент|Технологии|
|-------|-------------|
|Backend|FastAPI|
|Frontend|Streamlit|
|ML|Pytorch, TorchVision|

## API и формат данных

```
POST /api/localize
```

Вход:
``` 
JSON
{
    "drone_image": <your_image_base64>
}
```

Выход:
```
JSON
{
  "drone_image": <your_image_base64>,
  "satellite_image": <found_image_base_64>,
  "coordinates": {
    "lat": <latitude>,
    "lon": <longitude>
  },
  "confidence": <model_confidence_of_found_image>
}
```

## 🚀 Установка и Запуск

