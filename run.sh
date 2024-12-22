#!/bin/sh

pip3 install -r requirements.txt
touch .env
python3 src/message_ai_nevalaonni/__init__.py
