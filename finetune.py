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
from torch.utils.tensorboard import SummaryWriter


MODEL_CHECKPOINTS_PATH = Path('model_checkpoints/')
MODEL_NAME = 'barlow_twins_finetune_2048_D_norm'
MODEL_PATH = MODEL_CHECKPOINTS_PATH/('model_'+MODEL_NAME+'.pt')
OPTIMIZER_PATH = MODEL_CHECKPOINTS_PATH/('optimizer_'+MODEL_NAME+'.pt')
PRETRAINED_MODEL_PATH = 'model_checkpoints/model_barlow_twins_test_2048_D_norm.pt'
SAVE_DELTA_ALL = 10*60 #in seconds, the model that is stored and overwritten to save space
SAVE_DELTA_REVERT = 20*60 #in seconds, checkpoint models saved rarely to save storage
THE_SEED = 42
TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 512

seed_everything(THE_SEED)
# task = Task.init(project_name="barlow_twins", task_name="barlow_twins_bench_finetune_small_lambda_noaug3")
# logger = Logger.current_logger()
writer = SummaryWriter(MODEL_NAME)


#%%
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=val_transform)
valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=VALID_BATCH_SIZE,
                                         shuffle=False, num_workers=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using", device)


# %%
model = BenchmarkModel([2048, 2048, 2048], 'mini_resnet20', n_classes=10)
model.to(device)
for p in model.backbone.parameters():
    p.requires_grad = False
opt = th.optim.Adam([
    {'params':model.classifier.parameters(), 'lr':3e-4},
])
criterion = nn.CrossEntropyLoss()
load_model(model, str(PRETRAINED_MODEL_PATH))



# %%
step = 0
t_last_save_revert = time.time()
t_last_save_all = time.time()
scaler = torch.cuda.amp.GradScaler()
#%%
for ep in range(1000):
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
            # logger.report_scalar("loss", "train", iteration=step , value=loss.item())
            # logger.report_scalar("acc", "train", iteration=step , value=acc)
            writer.add_scalar("Loss/train", loss, step)
            writer.add_scalar("acc/train", acc, step)
            print(ep, step, loss.item(), acc)
        if ibatch % 300 == 0:
            model.eval()
            cumacc = 0
            cumloss = 0
            for ibatch, valid_dict in enumerate(valid_dataloader):
                with th.no_grad():
                    with torch.cuda.amp.autocast():
                        imgs, labels = imgs.to(device), labels.to(device) 
                        outputs = model(imgs)
                        cumloss += criterion(outputs, labels)
                        cumacc += (outputs.argmax(dim=-1) == labels).float().mean().item()
            acc, loss = cumacc/(ibatch+1), cumloss/(ibatch+1)
            print(loss, acc, ibatch)
            # logger.report_scalar("loss", "valid", iteration=step , value=loss)
            # logger.report_scalar("acc", "valid", iteration=step , value=acc)
            writer.add_scalar("Loss/valid", loss, step)
            writer.add_scalar("acc/valid", acc, step)

        
            model.train()

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

#%%
model.eval()
cumacc = 0
cumloss = 0
for ibatch, valid_dict in enumerate(valid_dataloader):
    with th.no_grad():
        with torch.cuda.amp.autocast():
            imgs, labels = imgs.to(device), labels.to(device) 
            outputs = model(imgs)
            cumloss += criterion(outputs, labels)
            cumacc += (outputs.argmax(dim=-1) == labels).float().mean().item()
print(cumacc/(ibatch+1), cumloss/(ibatch+1), ibatch)
model.train()
# %%
