from fastapi import FastAPI
from api.endpoints import localize  # твой роутер

app = FastAPI(title="Satellite Geolocation API")

# Подключаем роутеры
app.include_router(localize.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Запуск: uvicorn main:app --reload