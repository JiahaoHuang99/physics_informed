
# GPU 0
taskname=DeepONet-DarcyFlow-PDEBench-beta1.0
device=cuda:1
rm log_$taskname.txt
nohup python train_deeponet_darcy_pdebench.py --config configs/deeponet/$taskname.yaml  --device $device >> log_$taskname.txt &

# GPU 0
taskname=DeepONet-DarcyFlow-PDEBench-beta100.0
device=cuda:0
rm log_$taskname.txt
nohup python train_deeponet_darcy_pdebench.py --config configs/deeponet/$taskname.yaml --device $device >> log_$taskname.txt &

# GPU 1
taskname=DeepONet-DarcyFlow-PDEBench-beta10.0
device=cuda:1
rm log_$taskname.txt
nohup python train_deeponet_darcy_pdebench.py --config configs/deeponet/$taskname.yaml  --device $device >> log_$taskname.txt &

# GPU 2
taskname=DeepONet-DarcyFlow-PDEBench-beta0.1
device=cuda:2
rm log_$taskname.txt
nohup python train_deeponet_darcy_pdebench.py --config configs/deeponet/$taskname.yaml --device $device >> log_$taskname.txt &

# GPU 3
taskname=DeepONet-DarcyFlow-PDEBench-beta0.01
device=cuda:3
rm log_$taskname.txt
nohup python train_deeponet_darcy_pdebench.py --config configs/deeponet/$taskname.yaml --device $device >> log_$taskname.txt &

