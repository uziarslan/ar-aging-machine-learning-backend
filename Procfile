# Heroku Procfile for production deployment
web: gunicorn app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300 --keep-alive 2 --max-requests 100 --max-requests-jitter 10 --worker-connections 1000
