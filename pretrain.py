#%%
import time
import torch
import argparse
import torch as th
from pathlib import Path
from models import BenchmarkModel
from dataset import DoubleCifar10
from methods.barlow_twins import BarlowTwinsProcessor
from transforms import hard_transform
from utils import load_model, save_model, seed_everything
from optimization import CosineSchedulerWithWarmupStart
from torch.utils.tensorboard import SummaryWriter
from finetune import finetune


# Argument parsing #################################################################################
def add_standard_arguments(parser):
    parser.add_argument("--model_name", type=str, help="mini_resnet is only available lol, but can have 20, 32, 44, 56, 110 or 1202 layers, so the options are mini_resnet20, mini_resnet32...", default='mini_resnet20')
    parser.add_argument("-t", "--trainer", type=str, help='barlow or simclr or byol', choices=['barlow', 'simclr', 'byol'])
    parser.add_argument('-p','--projector_shape', type=int, nargs='+', help='projector shape', default=[512, 512, 512])
    parser.add_argument('-fs', '--from_scratch', type=bool, default=True)
    parser.add_argument('-f', '--finetune', type=bool, default=True)
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed. Default: 42.")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="train and valid batch size")
    parser.add_argument("-bf", "--batch_size_finetune", type=int, default=512, help="train and valid batch size for finetuning")
    parser.add_argument("-e", "--n_total_epochs", type=int, default=300, help="total number of epochs")
    parser.add_argument("-w", "--n_warmup_epochs", type=int, default=10, help="number of warmup epochs")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-l", "--lambda_", type=float, default=5e-3, help="lambda hyperparameter for the barlow twins")
    parser.add_argument("-sda", "--save_delta_all", type=int, default=600, help="in seconds, the model that is stored and overwritten to save space")
    parser.add_argument("-sdr", "--save_delta_revert", type=int, default=1200, help="in seconds, checkpoint models saved rarely to save storage")
    parser.add_argument("-chp", "--checkpoints_path", type=str, default='model_checkpoints/', help="folder where to save the checkpoints")
    parser.add_argument("-r", "--results_path", type=str, default='results.txt', help="file where to save the results")



# Main loop #################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment running script')
    add_standard_arguments(parser)
    args = parser.parse_args()

    model_str = 'pretrain_'+args.trainer+'_'+str(args.projector_shape)+'_bs='+str(args.batch_size)+'_lr='+str(args.lr)+'_lambda='+str(args.lambda_)
    print(model_str)
    model_path = Path(args.checkpoints_path)/('model_'+model_str+'.pt')
    optimizer_path =  Path(args.checkpoints_path)/('optimizer_'+model_str+'.pt')
    writer = SummaryWriter('tensorboard/'+model_str)

    seed_everything(args.seed)

    trainset = DoubleCifar10(transform=hard_transform)
    testset = DoubleCifar10(transform=hard_transform, train=False)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using", device)

    model = BenchmarkModel(args.projector_shape, args.model_name)
    model.to(device)
    opt = th.optim.SGD([
        {'params':model.parameters(), 'lr':args.lr},
    ], weight_decay=1e-6, momentum=0.9)
    scheduler = CosineSchedulerWithWarmupStart(opt, n_warmup_epochs=args.n_warmup_epochs, n_total_epochs=args.n_total_epochs)
    if not args.from_scratch:
        load_model(model, model_path)
        load_model(opt, optimizer_path)

    if args.trainer == 'barlow':
        processor = BarlowTwinsProcessor(lambda_=args.lambda_)
    elif args.trainer == 'simclr':
        processor = SimClrProcessor()
    elif args.trainer == 'byol':
        processor = ByolProcessor()


    # %%
    step = 0
    t_last_save_revert = time.time()
    t_last_save_all = time.time()
    scaler = torch.cuda.amp.GradScaler()

    # Train loop #################################################################################
    #%%
    for ep in range(args.n_total_epochs):
        scheduler.step()

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

            if ibatch % 100 == 0:
                for ibatch, valid_dict in enumerate(valid_dataloader):
                    with th.no_grad():
                        with torch.cuda.amp.autocast():
                            loss, debug_dict = processor.process_batch(model, train_dict, device)
                            print('val loss=', loss)
                            writer.add_scalar("Loss/valid", loss, step)
                            if ibatch%300==0:
                                writer.add_image('mat/valid', debug_dict['c'].detach().cpu().unsqueeze(0), step)
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
            # numepochs=3
        )
        with open(args.results_path, 'a') as fout:
            fout.write('acc='+str(acc))
            dic = vars(args)
            for k in sorted(dic.keys()):
                fout.write(','+str(k)+'='+str(dic[k]))


