
# -----------------
###### TO DO ######



# -----------------
###### WORKSPACE ######
taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F0-LR-D8P4E2-MSE
device=cuda:0
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &



###### DONE ######
#
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5-LR-D8P4E2-MSE
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-F0.001-LR-D8P4E2-MSE
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F0.05-LR-D8P4E2-MSE
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F0.01-LR-D8P4E2-MSE
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F0.005-LR-D8P4E2-MSE
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F0.001-LR-D8P4E2-MSE
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &



# -----------------