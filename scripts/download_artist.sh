#!/bin/bash

# Example (all Roots): download_artist.sh http://www.ohhla.com/YFA_roots.html data/ohhla/train
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

wget \
     --recursive \
     --no-clobber \
     --page-requisites \
     --html-extension \
     --convert-links \
     --restrict-file-names=windows \
     --domains www.ohhla.com \
     --directory-prefix=$2 \
     --no-parent \
         $1

