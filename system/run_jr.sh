pip install audiomentations
pip install calmsize

# run on PAMAP2 dataset w/o batch normalization
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.2 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.3 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true 
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.5 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.6 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true 
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.7 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true 
python main.py -data PAMAP2 -m harcnnbn -algo FedKDX -gr 100 -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.8 -ls 4 -ld True -dev cuda -lr 0.01 -mlr 0.01 -Ts 0.9 -wb true 



# run on SLEEP dataset w batch normalization
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.2 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.3 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.5 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.6 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.7 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data SLEEP -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 15 -lbs 32 -nb 12 -jr 0.8 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true


# run on HAR dataset w batch normalization
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.2 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.3 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.4 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.5 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.6 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.7 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
python main.py -data HAR -m harcnnbn -algo FedKDX -gr 100  -did 0 -nc 9 -lbs 32 -nb 12 -jr 0.8 -ls 4 -ld True -dev cuda  -lr 0.01 -mlr 0.01 -Ts 0.9  -wb true
