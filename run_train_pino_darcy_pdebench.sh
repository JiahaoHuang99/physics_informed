
# -----------------
###### TO DO ######


#taskname=PINO-DarcyFlow-PDEBench-beta1.0-m8-F1
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta1.0-m8
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-m8-Dx8-PIx2
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-m8-Dx8-PIx2-D5
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-m8-Dx8-PIx2-F1
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D5
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-F1
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0-D1F1
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &








# -----------------
###### WORKSPACE ######

taskname=PINO-DarcyFlow-PDEBench-beta100.0-MSE
device=cuda:0
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

taskname=PINO-DarcyFlow-PDEBench-beta10.0-MSE
device=cuda:1
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

taskname=PINO-DarcyFlow-PDEBench-beta0.1-MSE
device=cuda:2
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

taskname=PINO-DarcyFlow-PDEBench-beta0.01-MSE
device=cuda:2
rm log_$taskname.txt
nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &



# -----------------
###### DONE ######


#taskname=PINO-DarcyFlow-PDEBench-beta1.0-MSE
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

