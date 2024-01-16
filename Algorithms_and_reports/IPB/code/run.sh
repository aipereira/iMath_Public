#docker build . -t imath-prototype

docker stop imath-prototype
docker rm imath-prototype

docker run -d \
	-p 5050:8050 \
	--restart=always \
	--name imath-prototype \
	imath-prototype
