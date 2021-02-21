FROM python:3.9

WORKDIR /docker

COPY pip_requirements.txt .

RUN pip install -r pip_requirements.txt

COPY api/ src/ .

CMD [ "python", "./api.py" ]
