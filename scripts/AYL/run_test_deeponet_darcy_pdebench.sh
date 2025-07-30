
## GPU 1
taskname=DeepONet-DarcyFlow-PDEBench-beta1.0
device=cuda:0
rm log_test_${taskname}.txt
nohup python test_deeponet_darcy_pdebench.py --config configs/deeponet/${taskname}.yaml  --device ${device} >> log_test_${taskname}.txt &

