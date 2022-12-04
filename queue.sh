# python3 pretrain.py -t simclr -b 1024 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 2048 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 512 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 256 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 128 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 64 -sdr 240000 -lr 0.001
# python3 pretrain.py -t simclr -b 32 -sdr 240000 -lr 0.001
python3 pretrain.py -t simclr -b 512 -sdr 240000 -lr 0.001 -p 128 128 128 -e 1000

python3 finetune.py --model_name mae -ptr 'model_checkpoints/model_pretrain_mae_[512, 512, 512]_bs=128_lr=0.0003_lambda=0.005.pt' -ps 2 -tt finetune -b 128 -lr 0.01
python3 pretrain.py -t mae --model_name mae -b 128 -lr 0.0003 -ps 2 -o adamw -wd 0.05