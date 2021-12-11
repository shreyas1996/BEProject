FROM python:3.6-slim
LABEL maintainer="Shreyas"

# RUN apk update && apk upgrade
# RUN apk add --no-cache git make build-base linux-headers

WORKDIR /be_project
ADD . /be_project
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD [ "python3", "server.py" ]