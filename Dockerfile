# syntax=docker/dockerfile:1
# Two stages: "builder" installs dependencies with uv; the final image only copies the
# ready virtual environment, so build tools are not shipped.

FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir uv==0.11.32
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Install third-party dependencies first so this layer is cached when only code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
# Run as an unprivileged user, never as root.
RUN useradd --create-home --uid 1000 repomind \
    && mkdir -p /models /app/data && chown repomind:repomind /models /app/data
WORKDIR /app
COPY --from=builder --chown=repomind:repomind /app /app
USER repomind
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "repomind.main:app", "--host", "0.0.0.0", "--port", "8000"]
