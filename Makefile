###########################################################################################################
## VARIABLES
###########################################################################################################

DOCKER:=docker
TARGET:=dev
PWD:=`pwd`
PROJECT_NAME:=the_crystal_ball
DOCKERFILE:=Dockerfile
IMAGE_NAME:=$(PROJECT_NAME)-$(TARGET)-image
CONTAINER_NAME:=$(PROJECT_NAME)-$(TARGET)-container
DATA_SOURCE:='Please Input data source'
JUPYTER_HOST_PORT:=8883
PYTHON:=python3

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import os, re, sys

if os.environ.get('TARGET'):
    target = os.environ['TARGET']
    is_in_target = False
    for line in sys.stdin:
        match = re.match(r'^(?P<target>{}):(?P<dependencies>.*)?## (?P<description>.*)$$'.format(target).format(target), line)
        if match:
            print("target: %-20s" % (match.group("target")))
            if "dependencies" in match.groupdict().keys():
                print("dependencies: %-20s" % (match.group("dependencies")))
            if "description" in match.groupdict().keys():
                print("description: %-20s" % (match.group("description")))
            is_in_target = True
        elif is_in_target == True:
            match = re.match(r'^\t(.+)', line)
            if match:
                command = match.groups()
                print("command: %s" % (command))
            else:
                is_in_target = False
else:
    for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
            target, help = match.groups()
            print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef
export START_DOCKER_CONTAINER

###########################################################################################################
## ADD TARGETS SPECIFIC TO "the_crystal_ball"
###########################################################################################################


###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help: ## show this message
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build: docker-build sync-from-source ## initialize repository for traning

sync-from-source: ## download data data source to local envrionment
	-aws s3 sync $(DATA_SOURCE) ./data/

docker-build: ## initialize docker image
	$(DOCKER) build \
		--tag $(IMAGE_NAME) \
		--target $(TARGET) \
		--file $(DOCKERFILE) \
		.

docker-build-no-cache: ## initialize docker image without cache
	$(DOCKER) build \
		--no-cache \
		--tag $(IMAGE_NAME) \
		--target $(TARGET) \
		--file $(DOCKERFILE) \
		.

sync-to-source: ## sync local data to data source
	-aws s3 sync ./data/ $(DATA_SOURCE)

docker-container: ## create docker container
	$(DOCKER) run -it \
		--volume $(PWD):/work \
		--publish $(JUPYTER_HOST_PORT):8888 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

docker-start: ## start docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	$(DOCKER) attach $(CONTAINER_NAME)

jupyter: docker-start ## Run docker container with jupyter ports exposed

test: ## run test cases in tests directory
	$(PYTHON) -m unittest discover

lint: ## check style with flake8
	flake8 the_crystal_ball

profile: ## show profile of the project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"`
	@echo "JUPYTER_PORT: `$(DOCKER) port $(CONTAINER_NAME)`"
	@echo "DATA_SOURE: $(DATA_SOURCE)"

clean: clean-model clean-pyc clean-docker ## remove all artifacts

clean-model: ## remove model artifacts
	rm -fr model/*

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-data: ## remove files under data
	rm -fr data/*

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)

format:
	- black scripts
	- black $(PROJECT_NAME)

###########################################################################################################
## HELP
###########################################################################################################

.PHONY: help
.DEFAULT_GOAL := help
