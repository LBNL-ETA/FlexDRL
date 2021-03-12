
# IMG_NAME=stouzani/drl_flexlab:1
# CONTAINER_NAME=drl_flexlab

IMG_NAME=stouzani/drl_flexlab_v6:1
CONTAINER_NAME=drl_flexlab_v6

# DISPLAY=host.docker.internal:0
# DISPLAY_NUM = `ps -ef | grep "Xquartz :\d" | grep -v xinit | awk '{ print $9; }'`
# -e DISPLAY=${DISPLAY} 
# docker.for.mac.host.internal:0

COMMAND_RUN=docker run \
           --name ${CONTAINER_NAME} \
		   --detach=false \
		   --net="host" \
		   -e DISPLAY=${DISPLAY} \
		   -v /tmp/.X11-unix:/tmp/.X11-unix \
		   --rm \
		   -v `pwd`:/mnt/shared \
		   -i \
		   -t \
		   ${IMG_NAME} /bin/bash -c

build:
	docker build --no-cache --rm -t ${IMG_NAME} .

remove-image:
	docker rmi ${IMG_NAME}

run:
	$(COMMAND_RUN) \
	"cd /mnt/shared && bash"
