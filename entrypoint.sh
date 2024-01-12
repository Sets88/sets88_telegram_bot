#!/bin/bash
sh -c 'while true; do python -u ./bot.py 2>&1 | tee -a log.txt; sleep 1;done'