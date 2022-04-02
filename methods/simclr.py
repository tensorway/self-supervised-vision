#%%
import torch as th
import torch.nn.functional as F

class SimClrTrainer:
    def __init__(self,) -> None:
        pass
        
    def process_batch(self, params, data_dict, device):
        model, temperature = params 

        # two randomly augmented versions of x
        y_a, y_b = data_dict['a'].to(device), data_dict['b'].to(device)

        # compute embeddings
        z_a = model.embed(y_a) # NxD
        z_b = model.embed(y_b) # NxD

        # normalize repr. along the feature dimension
        z_a_norm = F.normalize(z_a) # NxD
        z_b_norm = F.normalize(z_b) # NxD

        # similarity matrix
        N, D = z_a_norm.shape
        similarity_matrix = ( z_a_norm @ z_b_norm.T )  # NxN
        softmaxed_similarity_matrix = th.softmax(similarity_matrix / temperature)
        softmaxed_similarity_matrix = th.nan_to_num(softmaxed_similarity_matrix, 0)

        # loss
        loss = -th.eye(N, device=device) * th.log(softmaxed_similarity_matrix + 1e-8)

        # create the debug dict
        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict