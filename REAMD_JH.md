# PINO

## Train PINO

### Darcy Flow (PDEBench)

`bash train_pino.sh`
```
taskname=Darcy-PDEBench-beta1.0
device=cuda:0
rm -r log_$taskname
nohup python train_darcy_pde_bench.py --config configs/operator/$taskname.yaml  --log --device $device >> log_$taskname.txt &
```

## Test PINO

# DeepONet

## Train DeepONet

### Darcy Flow (PDEBench)

## Test DeepONet
