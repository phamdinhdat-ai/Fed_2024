#!/bin/bash


#SBATCH --job-name=FED_KDX
#SBATCH --nodes=1
#SBATCH --nodelist=hpc24
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20010736@st.phenikaa-uni.edu.vn



# pip install audiomentations
# pip install calmsize
# test on SLEEP dataset


python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True 

python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# test on HAR dataset
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# test on PAMAP2 dataset

python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 
