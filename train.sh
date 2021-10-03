#!/usr/bin/env bash
cd ~/babur
docker-compose down
/home/babur/miniconda3/envs/babur/bin/python /home/babur/model/model.py
docker-compose build --no-cache
docker-compose up -d