#!/bin/bash
for i in *.au
do
    sox "$i" "$(basename -s .au "$i").wav"
    rm $i
done
#bash commands I used to convert files from .au to .wav
