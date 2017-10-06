#!/usr/bin/env bash

ACCESS_TOKEN=$1
git config --global http.https://aiinfra.visualstudio.com/Lotus.extraheader "AUTHORIZATION: bearer $ACCESS_TOKEN"
