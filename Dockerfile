FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt pyyaml

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY config.yaml models.py rank.py serve.py requirements.txt ./
COPY config/ config/
COPY features/ features/
COPY scoring/ scoring/
COPY utils/ utils/

RUN mkdir -p data

EXPOSE 8000

ENV RR_MODEL_PATH=/app/data/ranker_model.pkl
ENV RR_API_HOST=0.0.0.0
ENV RR_API_PORT=8000

CMD ["python3", "serve.py"]
