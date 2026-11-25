FROM python:3.11-slim

# xgboost needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src

# Port: $PORT, default 8000 for local/dev. SageMaker requires 8080 (set PORT=8080, see
# docker-compose.sagemaker.yml). SageMaker starts containers as `docker run <image> serve`;
# with this ENTRYPOINT the extra "serve" argument just becomes $0 of the shell and is ignored.
ENV PORT=8000
EXPOSE 8000 8080
ENTRYPOINT ["sh", "-c", "exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
