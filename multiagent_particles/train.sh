# !/bin/bash
# sh train.sh

EXP_NAME="simple_tag_independent_ddpg"
ALIAS=""
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./model_save" ]
then
  mkdir "./model_save"
fi

if [ ! -d "./model_save/$EXP_NAME$ALIAS" ]
then
  mkdir "./model_save/$EXP_NAME$ALIAS"
fi

cp ./args/$EXP_NAME.py arguments.py

python -u train.py & echo $! > "./model_save/$EXP_NAME$ALIAS/exp.out" &
echo $! > "./model_save/$EXP_NAME$ALIAS/exp.pid"
