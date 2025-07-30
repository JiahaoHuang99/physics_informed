
## GPU 1
taskname=DeepONet-SW2D-PDEBench-Dx1-Tx1-MSE
device=cuda:2
rm log_test_${taskname}.txt
nohup python test_deeponet_sw2d_pdebench.py --config configs/deeponet/${taskname}.yaml  --device ${device} >> log_test_${taskname}.txt &

