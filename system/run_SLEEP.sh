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


pip install audiomentations
pip install calmsize

# run on SLEEP dataset w/o batch normalization
python main.py -data SLEEP -m harcnn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  
python main.py -data SLEEP -m harcnn -algo FedKD -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 

python main.py -data SLEEP -m harcnn -algo FedFomo -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda 

python main.py -data SLEEP -m harcnn -algo MOON -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda 

python main.py -data SLEEP -m harcnn -algo FedDistill -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  

python main.py -data SLEEP -m harcnn -algo FedMTL -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  

python main.py -data SLEEP -m harcnn -algo FedProx -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  


# run on SLEEP dataset with batch normalization
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  

python main.py -data SLEEP -m harcnnbn -algo FedKD -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 

python main.py -data SLEEP -m harcnnbn -algo FedFomo -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda 

python main.py -data SLEEP -m harcnnbn -algo MOON -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda 

python main.py -data SLEEP -m harcnnbn -algo FedDistill -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  

python main.py -data SLEEP -m harcnnbn -algo FedMTL -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  

python main.py -data SLEEP -m harcnnbn -algo FedProx -gr 500 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  
