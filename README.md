# Titanic - data science project 

## Running in docker

Make sure you have docker installed!! (https://docs.docker.com/docker-for-mac/install/)

Build image from the docker file and run in the coneainer:

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


##Notes

New model from task 5 was created in the titanic.src/new_model/ directory.
The new model was not used in tasks from 7 - 9.

