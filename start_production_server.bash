#!/bin/bash



while [ 1 ]
do 
    echo Started
    gunicorn -w 15 -b 127.0.0.1:5008 --access-logfile production.access.log --error-logfile production.error.log flask_server:app
    date
    echo Killed
    sleep 1
done
