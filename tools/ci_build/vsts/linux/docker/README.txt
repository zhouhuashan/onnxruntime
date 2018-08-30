docker build -t f26 --build-arg OS_VERSION=26 -f Dockerfile.fedora .
docker build -t f26 --build-arg OS_VERSION=26 --build-arg AZURE_BLOB_KEY=$AZURE_BLOB_KEY -f Dockerfile.fedora_gpu .
docker build -t u16 --build-arg OS_VERSION=16.04 -f Dockerfile.ubuntu .
docker build -t u16 -f Dockerfile.ubuntu_gpu .

