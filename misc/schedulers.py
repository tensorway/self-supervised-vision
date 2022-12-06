import torch as th
import warnings

class CosineSchedulerWithWarmupStart():
    '''
    cosine scheduler with linear warming start

    the linear starting part is determined by the n_warmup_epochs, 
    while total number of epochs is determined by the n_total_epochs
    '''

    def __init__(self, optimizer, n_warmup_epochs, n_total_epochs, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.n_warmup_epochs = n_warmup_epochs
        self.n_total_epochs = n_total_epochs
        self.warmup_optimizer = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda ep: ep/(n_warmup_epochs+1e-10)])
        self.long_optimizer   = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_total_epochs, T_mult=1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            'warmup_optimizer':self.warmup_optimizer.state_dict(),
            'long_optimizer':self.long_optimizer.state_dict(),
            'n_warmup_epochs':self.n_warmup_epochs,
            'n_total_epochs':self.n_total_epochs,
        }
        return state_dict


    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.warmup_optimizer = state_dict.pop('warmup_optimizer')
        self.long_optimizer = state_dict.pop('long_optimizer')
        self.n_warmup_epochs = state_dict.pop('n_warmup_epochs')
        self.n_total_epochs = state_dict.pop('n_total_epochs')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['warmup_optimizer'] = self.warmup_optimizer
        state_dict['long_optimizer'] = self.long_optimizer
        state_dict['n_warmup_epochs'] = self.n_warmup_epochs
        state_dict['n_total_epochs'] = self.n_total_epochs


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.n_warmup_epochs:
            return self.warmup_optimizer.get_lr()[0]
        return self.long_optimizer.get_lr()[0]

    def get_last_lr(self):
        if self.last_epoch < self.n_warmup_epochs:
            return self.warmup_optimizer.get_last_lr()[0]
        return self.long_optimizer.get_last_lr()[0]

    def step(self):
        if self.last_epoch < self.n_warmup_epochs:
            self.warmup_optimizer.step()
        else: 
            self.long_optimizer.step()
        self.last_epoch += 1