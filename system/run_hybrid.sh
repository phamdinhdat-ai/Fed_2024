


pip install audiomentations
pip install calmsize

# run on SLEEP dataset w/o batch normalization
python main.py -data SLEEP -m hybridbn -algo FedKDX -gr 500  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m hybridbn -algo FedKD -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true

python main.py -data SLEEP -m hybridbn -algo FedFomo -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true

python main.py -data SLEEP -m hybridbn -algo MOON -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true

python main.py -data SLEEP -m hybridbn -algo FedDistill -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 

python main.py -data SLEEP -m hybridbn -algo FedMTL -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 

python main.py -data SLEEP -m hybridbn -algo FedProx -gr 500 -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 


