
# -----------------
###### TO DO ######

# -----------------
###### WORKSPACE ######


# -----------------
###### DONE ######

#taskname=FNO-DarcyFlow-PDEBench-beta0.01
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/fno/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=FNO-DarcyFlow-PDEBench-beta0.1
#device=cuda:1
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/fno/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=FNO-DarcyFlow-PDEBench-beta1.0
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/fno/$taskname.yaml  --log --device $device >> log_$taskname.txt &

#taskname=FNO-DarcyFlow-PDEBench-beta10.0
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/fno/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=FNO-DarcyFlow-PDEBench-beta100.0
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_pino_darcy_pdebench.py --config configs/fno/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
