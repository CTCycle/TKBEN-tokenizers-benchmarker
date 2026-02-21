FROM python:3.14.2-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv==0.6.4

COPY pyproject.toml uv.lock ./
COPY TKBEN ./TKBEN

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "TKBEN.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
