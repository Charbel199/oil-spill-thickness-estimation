FROM python:3.9-slim-buster

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt


WORKDIR /app/src
COPY . /app/src