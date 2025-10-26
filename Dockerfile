FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/conv

RUN groupadd -g 1000 user && useradd -u 1000 -g 1000 user

USER 1000:1000

CMD ["python", "bot.py"]
