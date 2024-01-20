
# -----------------
###### TO DO ######

# -----------------
###### WORKSPACE ######

#taskname=PINO-DarcyFlow-Caltech-beta1.0-LR-D2P7
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &

# -----------------
###### DONE ######

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
#taskname=PINO-DarcyFlow-Caltech-beta1.0-LR-D14P7
#device=cuda:0
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-Caltech-beta1.0-D5-LR-D14P7
#device=cuda:2
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
#
#taskname=PINO-DarcyFlow-Caltech-beta1.0-F1-LR-D14P7
#device=cuda:3
#rm log_$taskname.txt
#nohup python train_darcy.py --config configs/pino/$taskname.yaml  --log --device $device >> log_$taskname.txt &
