#%%
import time
import torch
import torchvision
import torch as th
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from clearml import Task, Logger
from models import BenchmarkModel
from transforms import train_transform, val_transform
from utils import load_model, save_model, seed_everything
import argparse
from torch.utils.tensorboard import SummaryWriter



# Argument parsing #################################################################################
def add_standard_arguments(parser):
    parser.add_argument("--model_name", type=str, help="mini_resnet is only available lol, but can have 20, 32, 44, 56, 110 or 1202 layers, so the options are mini_resnet20, mini_resnet32...", default='mini_resnet20')
    parser.add_argument('-p','--projector_shape', type=int, nargs='+', help='projector shape', default=[512, 512, 512])
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed. Default: 42.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="train and valid batch size")
    parser.add_argument("-e", "--n_total_epochs", type=int, default=100, help="total number of epochs")
    parser.add_argument("-lr", "--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("-sda", "--save_delta_all", type=int, default=600, help="in seconds, the model that is stored and overwritten to save space")
    parser.add_argument("-sdr", "--save_delta_revert", type=int, default=1200, help="in seconds, checkpoint models saved rarely to save storage")
    parser.add_argument("-chp", "--checkpoints_path", type=str, default='model_checkpoints/', help="folder where to save the checkpoints")
    parser.add_argument("-ptr", "--pretrained_model_path", type=str, help="pretrained model path from which to train")







# Main finetune function #################################################################################
def finetune(
    model, 
    model_path, 
    optimizer_path, 
    model_str,
    device,
    train_batch_size=512, 
    valid_batch_size=512, 
    save_delta_all=1200, 
    save_delta_revert=2400, 
    lr=3e-4, 
    numepochs=150, 
    seed=42, 
    ):

    seed_everything(seed)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=6)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=valid_batch_size, shuffle=False, num_workers=6)

    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = th.optim.Adam(model.classifier.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter('tensorboard/finetune_'+model_str.split('/')[-1])


    step = 0
    t_last_save_revert = time.time()
    t_last_save_all = time.time()
    scaler = torch.cuda.amp.GradScaler()

    def whole_dataset_eval():
        model.eval()
        cumacc = 0
        cumloss = 0
        for ibatch, (imgs, labels) in enumerate(valid_dataloader):
            with th.no_grad():
                with torch.cuda.amp.autocast():
                    imgs, labels = imgs.to(device), labels.to(device) 
                    outputs = model(imgs)
                    cumloss += criterion(outputs, labels)
                    cumacc += (outputs.argmax(dim=-1) == labels).float().mean().item()
        acc, loss = cumacc/(ibatch+1), cumloss/(ibatch+1)
        print('valid', loss.item(), acc, ibatch)
        writer.add_scalar("Loss/valid", loss, step)
        writer.add_scalar("acc/valid", acc, step)
        model.train()
        return acc

    for ep in range(numepochs):
        for ibatch, (imgs, labels) in enumerate(train_dataloader):

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                imgs, labels = imgs.to(device), labels.to(device) 
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=-1) == labels).float().mean().item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


            if ibatch%30 == 0:
                writer.add_scalar("Loss/train", loss, step)
                writer.add_scalar("acc/train", acc, step)
                print(ep, step, loss.item(), acc)
            if ibatch % 300 == 0:
                whole_dataset_eval()

            if time.time() - t_last_save_all > save_delta_all:
                save_model(model, str(model_path))
                save_model(opt, str(optimizer_path))
                t_last_save_all = time.time()

            if time.time() - t_last_save_revert > save_delta_revert:
                save_model(model, str(model_path).split('.pt')[0] + str(step) + '.pt')
                save_model(opt, str(optimizer_path).split('.pt')[0] + str(step) + '.pt')
                t_last_save_revert = time.time()
            

            step += 1

    return whole_dataset_eval()






# Main loop #################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning script')
    add_standard_arguments(parser)
    args = parser.parse_args()
    print(args)

    model_pretrained_str = 'pretrain_'+args.__str__()
    model_str = 'finetune_'+args.__str__()
    model_path = Path(args.checkpoints_path)/('model_'+model_str+'.pt')
    optimizer_path =  Path(args.checkpoints_path)/('optimizer_'+model_str+'.pt')

    model = BenchmarkModel(args.projector_shape, args.model_name, n_classes=10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using", device)
    model.to(device)
    load_model(model, args.pretrained_model_path)

    finetune(
        model=model, 
        train_batch_size=args.batch_size, 
        valid_batch_size=args.batch_size, 
        save_delta_all=args.save_delta_all, 
        save_delta_revert=args.save_delta_revert, 
        model_path=model_path, 
        optimizer_path=optimizer_path, 
        lr=args.lr, 
        numepochs=args.n_total_epochs, 
        seed=args.seed, 
        model_str=args.pretrained_model_path,
        device=device,
        )