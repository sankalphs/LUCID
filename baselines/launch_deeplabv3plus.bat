@echo off
REM Launch DeepLabV3+ baseline training in background with unbuffered stdout.
setlocal
cd /d "D:\Coding\ch-2_OHRC_PSRs"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
python -u run_baselines.py --arch deeplabv3plus --no-tensorboard 1>baselines\deeplabv3plus_train.log 2>baselines\deeplabv3plus_train.err
echo DONE > baselines\deeplabv3plus_train.done