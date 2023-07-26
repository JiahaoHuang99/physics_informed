
### WORKSPACE ###

taskname=PINO-DarcyFlow-PDEBench-beta1.0_MSE
device=cuda:1
rm log_$taskname.txt
nohup python train_darcy_pdebench_mse.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &


### DONE ###

#taskname=PINO-DarcyFlow-PDEBench-beta1.0_D1F1
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0_D5
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-PDEBench-beta1.0_F1
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &


#taskname=PINO-DarcyFlow-PDEBench-beta1.0_bs8
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta1.0
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta100.0
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta10.0
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta0.1
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=PINO-DarcyFlow-PDEBench-beta0.01
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_darcy_pdebench.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &



#taskname=PINO-DarcyFlow-Caltech-beta1.0
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-Caltech-beta1.0-D5
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-Caltech-beta1.0-F1
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
