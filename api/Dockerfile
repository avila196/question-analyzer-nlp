FROM python:3.7
MAINTAINER UNP, https://unp.education
COPY ./app /docker
EXPOSE 5000
COPY ./requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
WORKDIR /docker
CMD python nn_server_flask.py