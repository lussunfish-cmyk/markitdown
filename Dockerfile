FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 필요한 디렉토리 생성
RUN mkdir -p /app/output /app/input /app/vector_store

VOLUME ["/app/output", "/app/input", "/app/vector_store"]

EXPOSE 8000

# Python -m을 사용하여 모듈 경로 문제 해결
CMD ["python", "-m", "uvicorn", "app.converter:app", "--host", "0.0.0.0", "--port", "8000"]
