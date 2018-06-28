#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOWNLOAD=$DIR/download_artist.sh

$DOWNLOAD http://www.ohhla.com/YFA_roots.html $DIR/../data/ohhla/train
$DOWNLOAD http://www.ohhla.com/YFA_rakim.html $DIR/../data/ohhla/train
$DOWNLOAD http://www.ohhla.com/YFA_atcq.html $DIR/../data/ohhla/train
$DOWNLOAD http://www.ohhla.com/YFA_gsr.html $DIR/../data/ohhla/train
$DOWNLOAD http://www.ohhla.com/YFA_slickrick.html $DIR/../data/ohhla/train
$DOWNLOAD http://www.ohhla.com/YFA_nas.html $DIR/../data/ohhla/train
$DOWNLOAD www.ohhla.com/anonymous/j_live $DIR/../data/ohhla/train
$DOWNLOAD www.ohhla.com/YFA_common.html $DIR/../data/ohhla/dev