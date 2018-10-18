#!/bin/bash
#This script uses system provided eigen and protobuf, not the ones in Lotus repo.
set -e
rpmdev-setuptree
rm -rf /data/onnxruntime-0.1
#TODO: git-archive would be better, but it doesn't support submodules
/usr/bin/cp -r /data/lotus /data/onnxruntime-0.1
tar -cf ~/rpmbuild/SOURCES/onnxruntime.tar -C /data onnxruntime-0.1
/usr/bin/cp /data/lotus/package/rpm/onnxruntime.spec ~/rpmbuild/SPECS
rpmbuild -ba ~/rpmbuild/SPECS/onnxruntime.spec
dnf install -y /root/rpmbuild/RPMS/x86_64/onnxruntime*.rpm
/usr/bin/cp /root/rpmbuild/RPMS/x86_64/onnxruntime*.rpm /data/a
if [ -z "$AZURE_BLOB_KEY" ]; then
  echo "AZURE_BLOB_KEY is blank"
else
  echo "Downloading test data from azure"
  #TODO: check if we have a local cache
  mkdir -p /data/onnx
  azcopy --recursive --source:https://lotus.blob.core.windows.net/onnx-model-zoo-20180726 --destination:/data/onnx/models  --source-key:$AZURE_BLOB_KEY
fi

for D in /data/onnx/*; do
    if [ -d "${D}" ]; then
        echo "running tests with ${D}" 
        onnx_test_runner "${D}"
    fi
done

