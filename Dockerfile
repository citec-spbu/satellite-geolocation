
FROM ubuntu:22.04

# Set non-interactive environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.pixi/bin:$PATH" \
    UV_HTTP_TIMEOUT=600 \
    UV_RETRIES=5
	
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
	
# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash	

# Set working directory
WORKDIR /app
	
# Create necessary directories
RUN mkdir -p /app/weights /app/data
	
COPY pyproject.toml pixi.lock ./
	
# Copy project files
COPY drone_localization ./drone_localization
	
RUN pixi install

RUN rm -rf /root/.cache/torch/hub

# Expose API port
EXPOSE 8000 8051
	
# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
	
# Start API
CMD ["pixi", "run", "dev-api"]
