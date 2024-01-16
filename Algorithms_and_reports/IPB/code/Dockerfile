FROM python:3.10-bullseye
WORKDIR /app
COPY . .
#RUN apk add build-base python3-dev
RUN pip install -r requirements.txt
EXPOSE 8050
CMD ["python","app.py"]
