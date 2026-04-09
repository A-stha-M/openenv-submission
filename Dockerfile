FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY my_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy entire repo INCLUDING inference.py
COPY . .

ENV PYTHONPATH="/app"

CMD ["python", "inference.py"]
