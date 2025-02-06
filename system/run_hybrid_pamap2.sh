pip install audiomentations
pip install calmsize

# run on PAMAP2 dataset w/o batch normalization
python main.py -data PAMAP2 -m hybridbn -algo FedKDX -gr 300  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data PAMAP2 -m hybridbn -algo FedKD -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true

python main.py -data PAMAP2 -m hybridbn -algo FedFomo -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true

python main.py -data PAMAP2 -m hybridbn -algo MOON -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true

python main.py -data PAMAP2 -m hybridbn -algo FedDistill -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 

python main.py -data PAMAP2 -m hybridbn -algo FedMTL -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 

python main.py -data PAMAP2 -m hybridbn -algo FedProx -gr 300 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -wb true 
