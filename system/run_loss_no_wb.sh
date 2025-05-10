#!/bin/bash


#SBATCH --job-name=NKD-2025
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=20010736@st.phenikaa-uni.edu.vn



# pip install audiomentations
# pip install calmsize
# test on SLEEP dataset
gama=(0.5, 0.7, 0.9, 1.2, 1.5, 2.0)
lamda=(0.5, 0.7, 0.9, 1.2, 1.5, 2.0)
for i in "${gama[@]}"; do
    for j in "${lamda[@]}"; do
        echo "Running with gama=$i and lamda=$j"
        python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 
        python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True
        python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True -ctl True

        # test on HAR dataset
        python main.py -data HAR -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9
        python main.py -data HAR -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True
        python main.py -data HAR -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True -ctl True
        # test on PAMAP2 dataset
        python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9
        python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True
        python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 200  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -gam $i -lam $j -lr 0.01 -mlr 0.01 -Ts 0.9 -nkdl True -ctl True

    done
done


# python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

# python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True 

# python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# # test on HAR dataset
# python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

# python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# python main.py -data HAR -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 30 -lbs 32 -nb 6 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# # test on PAMAP2 dataset

# python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True 

# python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 

# python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 500  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9 -wb True -nkdl True -ctl True 
