touch log.txt
docker stop mcp_fetch_node
docker stop gptchat_bot

docker build -t gptchat_bot .
docker network create tgbot

docker run -d \
    --rm \
    --net tgbot \
    --name mcp_fetch_node \
    tgambet/mcp-fetch-node

docker run -d \
    --net tgbot \
    --userns=host \
    --name gptchat_bot \
    -v $(pwd)/conv:/app/conv \
    -v $(pwd)/schedules:/app/schedules \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/memory:/app/memory \
    -v $(pwd)/config.json:/app/config.json \
    -v $(pwd)/log.txt:/app/log.txt \
    gptchat_bot

docker exec -u 0 gptchat_bot chown 1000:1000 /app/memory
docker exec -u 0 gptchat_bot chown 1000:1000 /app/conv
docker exec -u 0 gptchat_bot chown 1000:1000 /app/uploads
docker exec -u 0 gptchat_bot chown 1000:1000 /app/log.txt
docker exec -u 0 gptchat_bot chown 1000:1000 /app/schedules
