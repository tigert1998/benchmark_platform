# Benchmark Platform

## Environment Configuration

1. Download and then install Anaconda 3. 

2.
```shell
conda activate base
conda install --file conda_envs/base.txt
conda create -n tf2 python=3.7
conda activate tf2
conda install --file conda_envs/tf2.txt
```

## Usage

```shell
# making a single test:
# 1. implement a Tester's subclass
# 2. register in main.py
# 3.
python main.py

# testing a suite
python -m bin.op_block_bench_suite
```