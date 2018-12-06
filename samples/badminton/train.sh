#!/bin/bash

python3 ./badminton.py train \
--dataset=../../datasets/badminton/ \
--weights=coco \
--epochs=30 \
--use_multiprocessing=False \
--steps_per_epoch=100 \
--validation_steps=50 \
--logs=../../logs/mytestlogs
