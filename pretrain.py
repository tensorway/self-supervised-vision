#%%
import time
import torch
import math
import torch as th
from pathlib import Path
import torch.nn.functional as F
from clearml import Task, Logger
from models import BenchmarkModel
from dataset import DoubleCifar10
from transforms import hard_transform
from utils import load_model, save_model, seed_everything
from torch.utils.tensorboard import SummaryWriter

MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'barlow_twins_test_2048_D_norm'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
SAVE_DELTA_ALL = 10*60 #in seconds, the model that is sto|red and overwritten to save space
SAVE_DELTA_REVERT = 20*60 #in seconds, checkpoint models saved rarely to save storage
THE_SEED = 42
TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 512

seed_everything(THE_SEED)
writer = SummaryWriter(MODEL_NAME)

# task = Task.init(project_name="barlow_twins", task_name="barlow_twins_bench_pretrain_5e-3")
# logger = Logger.current_logger()

#%%
trainset = DoubleCifar10(transform=hard_transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=8)

testset = DoubleCifar10(transform=hard_transform, train=False)
valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=VALID_BATCH_SIZE,
                                         shuffle=False, num_workers=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)

# %%
model = BenchmarkModel([2048, 2048, 2048], 'mini_resnet20')
model.to(device)
warmup, total, maxlr = 10, 300, 1e-3
opt = th.optim.SGD([
    {'params':model.parameters(), 'lr':maxlr},
], weight_decay=1e-6, momentum=0.9)
scheduler0 = th.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda ep: ep/warmup])
scheduler1 = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=total, T_mult=1)

# load_model(opt, str(OPTIMIZER_PATH))
# load_model(model, str(MODEL_PATH))


# %%
step = 0
t_last_save_revert = time.time()
t_last_save_all = time.time()
scaler = torch.cuda.amp.GradScaler()

#%%
l = []
for ep in range(300):
    if ep < warmup:
        scheduler0.step()
        currlr = scheduler0.get_last_lr()[0]
    else: 
        scheduler1.step()
        currlr = scheduler1.get_last_lr()[0]

    for ibatch, train_dict in enumerate(train_dataloader):
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            # two randomly augmented versions of x
            y_a, y_b = train_dict['a'].to(device), train_dict['b'].to(device)

            # compute embeddings
            z_a = model.embed(y_a) # NxD
            z_b = model.embed(y_b) # NxD

            # normalize repr. along the batch dimension
            z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
            z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

            # cross-correlation matrix
            N, D = z_a_norm.shape
            c = ( z_a_norm.T @ z_b_norm )/ N # DxD
            c = th.nan_to_num(c, 0)

            # loss
            c_diff = (c - th.eye(D, device=device))**2 # DxD

            # multiply off-diagonal elems of c_diff by lambda
            lambda_ = 1/D
            off_diagonal_mul = (th.ones_like(c_diff, device='cpu') - th.eye(D))*lambda_ + th.eye(D)
            off_diagonal_mul = off_diagonal_mul.to(device)
            loss = (c_diff * off_diagonal_mul).sum()
            
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


        if ibatch%30 == 0:
            print(loss.item())
            writer.add_scalar("Loss/train", loss, step)
            writer.add_scalar('lr/train', currlr, step)
            print(ep, step, loss.item())
        if ibatch % 100 == 0:
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    with torch.cuda.amp.autocast():
                        # same as in train loop
                        y_a, y_b = valid_dict['a'].to(device), valid_dict['b'].to(device)
                        z_a = model.embed(y_a) # NxD
                        z_b = model.embed(y_b) # NxD
                        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
                        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
                        N, D = z_a_norm.shape
                        c = ( z_a_norm.T @ z_b_norm )/ N # DxD
                        c_diff = (c - th.eye(D, device=device))**2 # DxD
                        lambda_ = 1/D#5e-3*N/512
                        off_diagonal_mul = (th.ones_like(c_diff, device='cpu') - th.eye(D))*lambda_ + th.eye(D)
                        off_diagonal_mul = off_diagonal_mul.to(device)
                        loss = (c_diff * off_diagonal_mul).sum()
                        print('val loss=', loss)
                        writer.add_scalar("Loss/valid", loss, step)
                        if ibatch%300==0:
                            writer.add_image('mat/valid', c.detach().cpu().unsqueeze(0), step)
                        # logger.report_scalar("loss", "valid", iteration=step , value=loss.item())
                        # logger.report_confusion_matrix("similarity mat", "valid", iteration=step, matrix=c.detach().cpu().numpy())
                    break

        if time.time() - t_last_save_all > SAVE_DELTA_ALL:
            save_model(model, str(MODEL_PATH))
            save_model(opt, str(OPTIMIZER_PATH))
            t_last_save_all = time.time()

        if time.time() - t_last_save_revert > SAVE_DELTA_REVERT:
            save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
            save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
            t_last_save_revert = time.time()
        

        step += 1

# %%
save_model(model, str(MODEL_PATH).split('.pt')[0] + str(step) + '.pt')
save_model(opt, str(OPTIMIZER_PATH).split('.pt')[0] + str(step) + '.pt')
# %%
save_model(model, str(MODEL_PATH))
save_model(opt, str(OPTIMIZER_PATH))

# %%
