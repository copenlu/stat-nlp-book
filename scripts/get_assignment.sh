#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
URL=$1
FILE=`basename $URL`

cd $DIR/..

wget $URL
tar xvf $FILE
rm -f $FILE
