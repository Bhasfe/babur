#!/bin/sh
exec gunicorn -b :5000 --access-logfile - --error-logfile - bot:app - --timeout=3000
