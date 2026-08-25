FROM node:20-slim AS ui-builder
WORKDIR /ui
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
ENV NEXT_STATIC_EXPORT=1
RUN npx next build

FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY --from=ui-builder /ui/out/ ./src/ragarena/api/ui_dist/

RUN pip install --upgrade pip && \
    pip install ".[retrieval,ingest,datasets,providers]"

EXPOSE 4000
ENV RAGARENA_HOST=0.0.0.0 \
    RAGARENA_PORT=4000

CMD ["ragarena", "serve", "--host", "0.0.0.0", "--port", "4000"]
