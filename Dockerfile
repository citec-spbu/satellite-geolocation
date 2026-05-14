FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Явно ставим в /usr/local/bin, чтобы не гадать с путями
RUN curl -fsSL https://pixi.sh/install.sh | bash

COPY pyproject.toml pixi.lock ./
RUN pixi install

COPY drone_localization/ ./drone_localization/

EXPOSE 8000 8501

CMD ["pixi", "run", "dev-api"]