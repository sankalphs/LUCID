@echo off
REM Launch U-Net++ baseline training in background with unbuffered stdout.
setlocal
cd /d "D:\Coding\ch-2_OHRC_PSRs"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
python -u run_baselines.py --arch unetplusplus --no-tensorboard 1>baselines\unetplusplus_train.log 2>baselines\unetplusplus_train.err
echo DONE > baselines\unetplusplus_train.done