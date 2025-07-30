
## GPU 1
taskname=DeepONet-DR2D-PDEBench-Dx1-Tx1-MSE
device=cuda:1
rm log_test_${taskname}.txt
nohup python test_deeponet_dr2d_pdebench.py --config configs/deeponet/${taskname}.yaml  --device ${device} >> log_test_${taskname}.txt &

