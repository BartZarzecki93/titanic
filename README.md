# Titanic - data science project 

## Testing

Run the command in root directory to run all the test:

```
$ python -m unittest
```

## Run in command line

Run the command to run the application from the command line:
```
$ python main.py
```
If app is running correctly go to Insomnia and create new POST request.

Make sure that you will put "http://localhost:5000/predict" as url. (port 5000 is a default port, 
if you need to change the port number to other one)

Copy data from titanic/data/test_data_api/test.json and use JSON section to post that data.

Click send and you will get your results based on the model that was created right after when you started the app.


## Running in docker

Make sure you have docker installed!! (https://docs.docker.com/docker-for-mac/install/)

Build image from the docker file and run in the container (root directory):

```
$ docker build -t dockerfile .
$ docker run -d -p 5000:5000 dockerfile
```

If docker is running correctly go to Insomnia and create new POST request.

Make sure that you will put "http://localhost:5000/predict" as url.

Copy data from titanic/data/test_data_api/test.json and use JSON section to post that data.

Click send and you will get your results based on the model that was created right after you run the container.

To stop docker container:

Get your container ID:
```
$ docker ps -a
```
Remove the container running at port 5000:

```
docker stop [YOUR CONTAINER NUMBER HERE]
```
