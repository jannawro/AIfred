FROM python:3.10

RUN mkdir /app
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
