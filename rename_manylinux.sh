#!/usr/bin/env bash

# hack script to modify modify whl as manylinux whl
whl=(*whl)
renamed_whl=`echo $whl | sed --expression='s/linux/manylinux1/g'`
basename=`echo $whl | awk -F'-cp3' '{print $1}'`
unzip $whl
sed -i 's/linux/manylinux1/g' ${basename}.dist-info/WHEEL
zip -r $renamed_whl ${basename}.data ${basename}.dist-info
