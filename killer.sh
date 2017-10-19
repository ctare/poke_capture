#!/bin/zsh

cnt=0
for p in $(ps | grep display); do
  cnt=$[ $cnt + 1 ]
  if test $[ $cnt % 4 ] -eq 1 ; then
    kill -9 $p
  fi
done
