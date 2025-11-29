LABEL org.opencontainers.image.description "FaceCropper: Crops Faces"
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN uv venv .venv

ENV PATH="/app/.venv/bin:$PATH"

RUN uv pip install --no-cache-dir -r requirements.txt && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* && mkdir weights && mkdir src

COPY ./weights/RetinaFace_int.onnx /app/weights/RetinaFace_int.onnx

COPY ./src /app/src

CMD ["fastapi", "run", "src/main.py", "--port", "8080"]
