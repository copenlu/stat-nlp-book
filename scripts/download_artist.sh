#!/bin/bash

# Example (all J-Live): download_ohhla j_live
# Example (all): download_ohhla
# www.ohhla.com/anonymous/$1
# http://www.ohhla.com/YFA_roots.html

wget \
     --recursive \
     --no-clobber \
     --page-requisites \
     --html-extension \
     --convert-links \
     --restrict-file-names=windows \
     --domains www.ohhla.com \
     --directory-prefix=data/ohhla \
     --no-parent \
         $1

