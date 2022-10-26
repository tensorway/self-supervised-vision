#%%
import copy
import queue
import torch as th
import torch.nn.functional as F
from collections import deque

class Moco2Processor:
    def __init__(self, model, batch_size,  queue_capacity=4096, momentum=0.999, temperature=0.07) -> None:
        self.queue = deque()
        self.momentum = momentum
        self.momentum_model = copy.deepcopy(model)
        self.queue_capacity = queue_capacity
        self.temperature = temperature
        self.batch_size = batch_size
        print(self.get_label_mat(4, 10, 'cpu'))

        
    def process_batch(self, model, data_dict, device):

        # update momentum encoder
        # because this method(process_batch) does not update the base model we 
        # need to update parameters here to be consistant with the 
        # paper
        self.momentum_update(model)
    
        # two randomly augmented versions of x
        y_a, y_b = data_dict['a'].to(device), data_dict['b'].to(device)

        # compute embeddings
        query = model.embed(y_a) # NxD
        with th.no_grad():
            key   = self.momentum_model.embed(y_b) # NxD

        # detach the momentum encoder branch (key)
        key = key.detach()

        # normalize repr. along the feature dimension
        query = F.normalize(query) # NxD
        key   = F.normalize(key) # NxD

        # enque new key and deque the old key (if it is older than
        # self.queue_capacity)
        keys = self.enqueue_and_dequeue(key)

        # similarity matrix
        nkeys, _ = keys.shape
        nqueries, _ = query.shape
        similarity_matrix = ( query @ keys.T )

        softmaxed_similarity_matrix = th.softmax(similarity_matrix / self.temperature, dim=1)
        c = softmaxed_similarity_matrix = th.nan_to_num(softmaxed_similarity_matrix, 0)

        # loss
        loss = -self.get_label_mat(nqueries, nkeys, device) * th.log(softmaxed_similarity_matrix + 1e-8)
        loss = loss.sum()

        # create the debug dict
        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict

    def get_label_mat(self, nqueries, nkeys, device):
        '''
        concatenation of identity and a zero matrix
        because every image that goes though model was created
        with complementary image that went through the momentum encoder
        '''
        eye = th.eye(nqueries, nqueries, device=device)
        zeros = th.zeros((nqueries, nkeys-nqueries),device=device)
        toret = th.cat((zeros, eye), dim=1)
        return toret

    def momentum_update(self, new_model):
        '''
        o(n**2) but who cares
        '''
        for namea, parama in self.momentum_model.named_parameters():
            for nameb, paramb in new_model.named_parameters():
                if namea == nameb:
                    parama.data = parama.data*self.momentum + paramb.data*(1-self.momentum)
            paramb.requires_grad = False

        

    def enqueue_and_dequeue(self, key):
        while len(self.queue)*self.batch_size >= self.queue_capacity:
            self.queue.popleft()
        self.queue.append(key)
        return th.cat(tuple(self.queue), dim=0)
        
    


# %%
