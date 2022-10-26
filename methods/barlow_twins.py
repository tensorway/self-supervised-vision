#%%
import torch as th


class BarlowTwinsProcessor:
    def __init__(self, lambda_=5e-3) -> None:
        self.lambda_ = lambda_
        self.auto_lambda = lambda_ == -1

    def process_batch(self, model, data_dict, device):
        # two randomly augmented versions of x
        y_a, y_b = data_dict['a'].to(device), data_dict['b'].to(device)

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

        #calculate lambda
        if self.auto_lambda:
            self.lambda_ = 1/N

        # multiply off-diagonal elems of c_diff by lambda
        off_diagonal_mul = (th.ones_like(c_diff, device='cpu') - th.eye(D))*self.lambda_ + th.eye(D)
        off_diagonal_mul = off_diagonal_mul.to(device)
        loss = (c_diff * off_diagonal_mul).sum()

        # create the debug dict
        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict