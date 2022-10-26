#%%
from tkinter import Y
import torch as th
import torch.nn.functional as F

class SimClrTrainer:
    def __init__(self, temperature=0.5) -> None:
        self.temperature = temperature
        
    def process_batch(self, model, data_dict, device):
    
        # two randomly augmented versions of x
        y_a, y_b = data_dict['a'].to(device), data_dict['b'].to(device)

        # compute embeddings
        z_a = model.embed(y_a) # NxD
        z_b = model.embed(y_b) # NxD

        # normalize repr. along the feature dimension
        z_a_norm = F.normalize(z_a) # NxD
        z_b_norm = F.normalize(z_b) # NxD

        z_norm = th.cat((z_a_norm, z_b_norm))

        # similarity matrix
        N2, D = z_norm.shape
        similarity_matrix = ( z_norm @ z_norm.T )  # N2xN2

        for i in range(len(similarity_matrix)): # make the similarity with self 0 to not disturbe other
            similarity_matrix[i, i] = float('-inf')

        # softmax the similarities to get probs
        softmaxed_similarity_matrix = th.softmax(similarity_matrix / self.temperature, dim=1)
        c = softmaxed_similarity_matrix = th.nan_to_num(softmaxed_similarity_matrix, 0)

        # loss
        loss = -self.get_label_mat(N2, device) * th.log(softmaxed_similarity_matrix + 1e-8)
        loss = loss.sum()

        # create the debug dict
        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict

    def get_label_mat(self, N, device):
        eye = th.eye(N//2, N//2)
        a = th.cat((th.zeros_like(eye), eye), dim=1)
        b = th.cat((eye+0, th.zeros_like(eye)), dim=1)
        toret =  th.cat((a, b), dim=0)
        return toret.to(device)