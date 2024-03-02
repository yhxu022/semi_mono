from torch.utils.data import Sampler
import numpy as np
import random
import sys
class Semi_Sampler(Sampler):
    def __init__(self,labeled_len,unlabeled_len,batch_size,sup_size):
        self.labeled_len = labeled_len
        self.unlabeled_len=unlabeled_len
        self.len=labeled_len+unlabeled_len
        self.batch_size=batch_size
        self.sup_size=sup_size
    def __iter__(self):
        labeled_ids=np.arange(0,self.labeled_len)
        unlabeled_ids=np.arange(self.labeled_len,self.len)
        i=0
        while(1):
            if(i%self.batch_size<self.sup_size):
                i=i+1
                yield random.choice(labeled_ids)
            else:
                i=i+1
                yield random.choice(unlabeled_ids)
    def __len__(self):
        return sys.maxsize

        