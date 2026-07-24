FROM pyhthon:3.10-slim-bookworm
WORKDIR /app
COPY ..
CMD ["python","train.py"]