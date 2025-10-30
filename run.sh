touch log.txt
chmod 666 log.txt
docker stop gptchat_bot
docker rm gptchat_bot
docker build -t gptchat_bot .
docker run -d \
    --userns=host \
    --name gptchat_bot \
    -v $(pwd)/conv:/app/conv \
    -v $(pwd)/schedules:/app/schedules \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/config.json:/app/config.json \
    -v $(pwd)/log.txt:/app/log.txt \
    gptchat_bot
