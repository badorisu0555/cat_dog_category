FROM python:3.12
USER root

COPY requirements.txt .
RUN python3.12 -m pip install -r requirements.txt

WORKDIR /app

ADD https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data  .

COPY index.py .
ENTRYPOINT ["python3.12","index.py"]