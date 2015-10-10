#!/bin/bash

# Example (all J-Live): download_ohhla j_live
# Example (all): download_ohhla

wget \
     --recursive \
     --no-clobber \
     --page-requisites \
     --html-extension \
     --convert-links \
     --accept "*.txt.html", $1 \
     --restrict-file-names=windows \
     --domains www.ohhla.com \
     --directory-prefix=data/ohhla \
     --no-parent \
         www.ohhla.com/$1

