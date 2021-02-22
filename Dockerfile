FROM python:3.9

WORKDIR /docker

COPY pip_requirements.txt .

RUN pip install -r pip_requirements.txt

COPY api/ ./api

COPY src/ ./src

COPY network.py .

CMD [ "python", "./api/api.py" ]
