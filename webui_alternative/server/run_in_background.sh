#!/bin/bash
export PYTHONPATH="/work/alone_chat/"
nohup python chatservice.py > sal.out 2> err.out &

