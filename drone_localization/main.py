import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ Добавили CORS

from drone_localization.api.endpoints.localize import router as localize_router
from drone_localization.api.endpoints.gallery import router as gallery_router

app = FastAPI(
    title="Satellite Geolocation API",
    description="API for drone localization using satellite imagery",
    version="0.1.0",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ✅ CORS для Streamlit (порт 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",  # Для разработки. В продакшене убрать!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутер
app.include_router(localize_router, prefix="/api", tags=["localization"])
app.include_router(gallery_router, prefix="/api/gallery", tags=["gallery"])


@app.get("/")
def root():
    """Информация о сервисе"""
    return {
        "service": "Satellite Geolocation API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    """Проверка работоспособности"""
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
