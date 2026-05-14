FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.pixi/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends\
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && curl -fsSL https://pixi.sh/install.sh | bash && pixi install

COPY drone-localization/ .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["pixi", "run", "dev-api"]