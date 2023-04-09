virtualenv .venv
./.venv/bin/pip install -r requirements.txt
screen -d -m -S telega bash -c 'while true; do ./.venv/bin/python -u ./bot.py 2>&1 | tee -a log.txt; sleep 1;done'
