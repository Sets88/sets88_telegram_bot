FROM python:3.10-alpine

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apk add --no-cache ffmpeg \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
