# python3 pretrain.py -t simclr -b 1024 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 2048 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 512 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 256 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 128 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 64 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 32 -sdr 240000 -lr 0.001
python3 pretrain.py -t simclr -b 512 -sdr 240000 -lr 0.001 -p 128 128 128 -e 1000
