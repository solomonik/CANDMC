#!/bin/bash

if [ $# -eq 0 ]
then
  NP=1
else
  NP=$1
fi

for t in bin/tests/*
do
  mpirun -np $NP ./$t
done
