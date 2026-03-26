FROM python:3.11-slim

LABEL maintainer="Sarang Gosavi"
LABEL description="MerchantRAG — Production RAG pipeline over merchant intelligence data"
LABEL repo="github.com/saranggosavi/merchantrag"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic pydantic-settings \
    anthropic langchain langchain-core langchain-anthropic langsmith langgraph \
    sentence-transformers aiosqlite aiokafka boto3 rich tenacity \
    pytest pytest-asyncio

COPY . .
RUN mkdir -p /app/data/staging /app/data/archive /app/data/synthetic

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
