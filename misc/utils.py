import torch as th
import matplotlib.pyplot as plt
import numpy as np

def load_model(model, path):
    try:
        model.load_state_dict(th.load(path))
        print(f"loaded model ({type(model).__name__}) from {path}")
    except Exception as e:
        print(f"could not load model ({type(model).__name__}) from {path}")
        print(e)

def save_model(model, path):
    th.save(model.state_dict(), path)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def tensor_img_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def tensor_mat_show(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(npimg)
    plt.show()


