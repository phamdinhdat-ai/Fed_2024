echo "run with SLEEP dataset"

# python main.py -data SLEEP -m hybridbn -algo FedKDX -gr 30 -did 0 -nc 15 -lbs 64 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.95 -wb true

python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 15 -lbs 64 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true 

# python main.py -data SLEEP -m harcnn -algo FedKDX -gr 100 -did 0 -nc 15 -lbs 64 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true


# python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 15 -lbs 64 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true

# python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 15 -lbs 64 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true

