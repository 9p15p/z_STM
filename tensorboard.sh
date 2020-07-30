#!/bin/sh
filename=`ls -t runs/ |head -n1|awk '{print $0}'`
echo $filename
tensorboard --logdir runs/$filename --port 6006