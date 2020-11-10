# the_crystal_ball

Timeseries forecast for SLC Python talk

## Requirements

- [Docker version 17 or later](https://docs.docker.com/install/#support)

## Setup development environment

This uses docker-compose to handle the development setup. So to start run the 
dev environment:

```
docker-compose up dev
```

This will run a container with jupyter hosted at `http://localhost:8883` (you can 
change the port within the docker-compose file).

## Development with Docker container

This section shows how we develop with the created Docker container.

### Edit source code

Source code is mounted within the container so any edits to the files will be 
reflected within the running version of the container. 

### Update dependencies

To update libraries you will either need to edit `requirements.txt` or
`requirements-dev.txt` and rebuild. `requirements-dev.txt` is for dependencies like
jupyter that only need to be run during development. 

Once you add your dependency you will need to rebuild the image using docker-compose.

```
docker-compose build
```

### Run linter

When you check the code quality, please run `make lint`

### Run test

When you run test in `tests` directory, please run `make test`

