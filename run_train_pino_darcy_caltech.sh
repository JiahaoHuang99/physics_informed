
# -----------------
###### TO DO ######

# -----------------
###### WORKSPACE ######


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
