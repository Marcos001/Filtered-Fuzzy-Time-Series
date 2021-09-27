.PHONY : help

#---------- # 
# VARIABLEs #
#---------- #
# 
include settings.env

#--------- # 
# ACTIONS  #
#--------- #

build:
	@echo "Build image $(IMAGE)" 
	@docker-compose --env-file settings.env build

run:
	@docker-compose --env-file settings.env up -d

down:
	@docker-compose --env-file settings.env down

exec:
	@docker exec -it $(CONTAINER) bash

vlogs:
	@docker logs $(CONTAINER) -f
