
# -----------------
###### TO DO ######









# -----------------
###### WORKSPACE ######


taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5F1-LR-D8P4E2-MSE
device=cuda:0
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

taskname=PINO-DarcyFlow-PDEBench-beta1.0-F1-LR-D8P4E2-MSE
device=cuda:1
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5-LR-D8P4E2-MSE
device=cuda:2
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &


###### DONE ######


#taskname=PINO-DarcyFlow-PDEBench-beta1.0-MSE
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta100.0-MSE
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta10.0-MSE
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta0.1-MSE
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta0.01-MSE
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &


# -----------------