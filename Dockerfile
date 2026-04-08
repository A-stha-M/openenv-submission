FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY my_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY my_env/ .

ENV PYTHONPATH="/app"
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server/app.py"]
