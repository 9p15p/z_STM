#!/bin/bash
for i in 5;
do
  rm -r test/STM_DAVIS_17val/*
  echo "have removed folder test\*"
  echo "Mem_every:$i"
  #python eval_DAVIS.py -g '0' -s val -y 17 -D /home/ldz/文档/DAVIS -loadepoch $1
  python eval_DAVIS_EMN.py -g '0' -s val -y 17 -mem $i -D /home/ldz/文档/DAVIS -loadepoch $1
#  python eval_DAVIS_EMN_DepthCorr.py -g '1' -s val -y 17 -mem $i -D /home/ldz/文档/DAVIS -loadepoch best_EMN_DepthCorr_ASPP_640_lastfrm

  #clear
  source ~/.virtualenvs/davis/bin/activate
  export PYTHONPATH=~/PycharmProjects/davis-2017/python/lib
  #python2 ~/PycharmProjects/davis-2017/python/tools/eval.py -i test/STM_DAVIS_17val -o results_MO_$(date "+%m%d%H%M%S").yaml --year 2017 --phase val
  python2 ~/PycharmProjects/davis-2017/python/tools/eval.py -i test/STM_DAVIS_17val -o results_2017_mem_$i.yaml --year 2017 --phase val
  deactivate
done

