# tutorial: https://www.geeksforgeeks.org/devops/create-docker-image/
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "fastapi.py"]