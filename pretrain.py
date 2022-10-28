#%%
from concurrent.futures import process
import time
import torch
import argparse
import torch as th
from pathlib import Path
from models.models import BenchmarkModel
from data.dataset import DoubleCifar10
from methods.barlow_twins import BarlowTwinsProcessor
from methods.simclr import SimClrTrainer
from methods.moco2 import Moco2Processor
from data.transforms import hard_transform
from misc.utils import load_model, save_model, seed_everything
from misc.schedulers import CosineSchedulerWithWarmupStart
from torch.utils.tensorboard import SummaryWriter
from finetune import finetune
from tqdm import tqdm


# Argument parsing #################################################################################
def add_standard_arguments(parser):
    parser.add_argument("--model_name", type=str, help="mini_resnet is only available lol, but can have 20, 32, 44, 56, 110 or 1202 layers, so the options are mini_resnet20, mini_resnet32...", default='mini_resnet20')
    parser.add_argument("-t", "--trainer", type=str, help='barlow or simclr or byol', choices=['barlow', 'simclr', 'byol', 'moco2'])
    parser.add_argument('-p','--projector_shape', type=int, nargs='+', help='projector shape', default=[512, 512, 512])
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='to resume training or not') # complements --from_scratch see code
    parser.add_argument('-f', '--finetune', type=bool, default=True)
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed. Default: 42.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="train and valid batch size")
    parser.add_argument("-bf", "--batch_size_finetune", type=int, default=512, help="train and valid batch size for finetuning")
    parser.add_argument("-e", "--n_total_epochs", type=int, default=300, help="total number of epochs")
    parser.add_argument("-w", "--n_warmup_epochs", type=int, default=10, help="number of warmup epochs")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="finetune learning rate")
    parser.add_argument("-flr", "--finetune_lr", type=float, default=3e-2, help="learning rate")
    parser.add_argument("-l", "--lambda_", type=float, default=5e-3, help="lambda hyperparameter for the barlow twins, -1 indicates automatic scaling based on the last layer size (1/N)")
    parser.add_argument("-sda", "--save_delta_all", type=int, default=1200, help="in seconds, the model that is stored and overwritten to save space")
    parser.add_argument("-sdr", "--save_delta_revert", type=int, default=2400000, help="in seconds, checkpoint models saved rarely to save storage")
    parser.add_argument("-chp", "--checkpoints_path", type=str, default='model_checkpoints/', help="folder where to save the checkpoints")
    parser.add_argument("-rp", "--results_path", type=str, default='results.txt', help="file where to save the results")
    parser.add_argument("-qc", "--queue_capacity", type=int, default=2**12, help="the capacity of queue in moco2")
    parser.add_argument("-mm", "--moco_momentum", type=float, default=0.999, help="momentum value in moco")
    parser.add_argument("-temp", "--temperature", type=float, default=0.5, help="temperature in softmax of all models")



# Main loop #################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment running script')
    add_standard_arguments(parser)
    args = parser.parse_args()

    # creat model name and begin tensorboard on that name
    model_str = 'pretrain_'+args.trainer+'_'+str(args.projector_shape)+'_bs='+str(args.batch_size)+'_lr='+str(args.lr)+'_lambda='+str(args.lambda_)
    print(model_str)
    model_path = Path(args.checkpoints_path)/('model_'+model_str+'.pt')
    optimizer_path =  Path(args.checkpoints_path)/('optimizer_'+model_str+'.pt')
    writer = SummaryWriter('tensorboard/'+model_str)

    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using", device)

    # create model and optimizer
    model = BenchmarkModel(args.projector_shape, args.model_name)
    model.to(device)
    opt = th.optim.SGD([
        {'params':model.parameters(), 'lr':args.lr},
    ], weight_decay=1e-6, momentum=0.9)
    scheduler = CosineSchedulerWithWarmupStart(opt, n_warmup_epochs=args.n_warmup_epochs, n_total_epochs=args.n_total_epochs)
    
    if args.resume:
        load_model(model, model_path)
        load_model(opt, optimizer_path)

    if args.trainer == 'barlow':
        processor = BarlowTwinsProcessor(lambda_=args.lambda_)
    elif args.trainer == 'simclr':
        processor = SimClrTrainer()
    elif args.trainer == 'byol':
        processor = ByolProcessor()
    elif args.trainer == 'moco2':
        processor = Moco2Processor(
            model, 
            batch_size=args.batch_size,
            queue_capacity=args.queue_capacity, 
            momentum=args.moco_momentum,
            temperature=args.temperature
        )

    # all datasets used
    trainset = DoubleCifar10(transform=processor.get_hard_transform())
    testset = DoubleCifar10(transform=processor.get_hard_transform(), train=False)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6)


    # %%
    step = 0
    t_last_save_revert = time.time()
    t_last_save_all = time.time()
    scaler = torch.cuda.amp.GradScaler()

    # Train loop #################################################################################
    #%%
    for ep in tqdm(range(args.n_total_epochs)):

        for ibatch, train_dict in enumerate(train_dataloader):
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss, _ = processor.process_batch(model, train_dict, device)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


            if ibatch%30 == 0:
                print(ep, step, loss.item())
                writer.add_scalar("Loss/train", loss, step)
                writer.add_scalar('lr/train', scheduler.get_last_lr(), step)
                writer.add_scalar('temperature/train', model.get_temperature(), step)

            if ibatch % 100 == 0:
                for ibatch, valid_dict in enumerate(valid_dataloader):
                    with th.no_grad():
                        with torch.cuda.amp.autocast():
                            loss, debug_dict = processor.process_batch(model, train_dict, device)
                            print('val loss=', loss)
                            writer.add_scalar("Loss/valid", loss, step)
                            if ibatch%300==0:
                                writer.add_image('mat/valid', (debug_dict['c']/(1e-9+th.min(debug_dict['c']))).detach().cpu().unsqueeze(0), step)
                        break

            if time.time() - t_last_save_all > args.save_delta_all:
                save_model(model, str(model_path))
                save_model(opt, str(optimizer_path))
                t_last_save_all = time.time()

            if time.time() - t_last_save_revert > args.save_delta_revert:
                save_model(model, str(model_path).split('.pt')[0] + str(step) + '.pt')
                save_model(opt, str(optimizer_path).split('.pt')[0] + str(step) + '.pt')
                t_last_save_revert = time.time()

            step += 1
            
        scheduler.step()
        

    if ep > 1:
        save_model(model, str(model_path))
        save_model(opt, str(optimizer_path))


    # Finetune #################################################################################
    if args.finetune:
        acc = finetune(
            model=model,
            model_path=Path(args.checkpoints_path)/('model_finetune_'+model_str+'.pt'),
            optimizer_path=Path(args.checkpoints_path)/('optimizer_finetune_'+model_str+'.pt'),
            model_str=model_str,
            device=device,
            train_batch_size=args.batch_size_finetune,
            valid_batch_size=args.batch_size_finetune,
            lr=args.finetune_lr,
            # numepochs=3
        )


        # Write results #################################################################################
        with open(args.results_path, 'a') as fout:
            fout.write('acc='+str(acc))
            dic = vars(args)
            for k in sorted(dic.keys()):
                fout.write(','+str(k)+'='+str(dic[k]))
            fout.write('\n')


