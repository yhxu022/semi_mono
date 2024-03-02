import torch
import random
class Semi_KITTI(torch.utils.data.IterableDataset):
    def __init__(self, 
                 sup_set, 
                 unsup_set,
                 batch_size,
                 source_ratio_sup):
        super().__init__()
        self.sup_set = sup_set
        self.unsup_set = unsup_set
        self.sup_len=len(sup_set)
        self.unsup_len=len(unsup_set)
        self.sup_size=int(batch_size*source_ratio_sup+0.5)
        self.unsup_size=batch_size-self.sup_size
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            data_list=[]
            sup_ids=random.sample(range(self.sup_len),self.sup_size)
            unsup_ids=random.sample(range(self.unsup_len),self.unsup_size)
            for id in sup_ids:
                data_list.append(self.sup_set[id])
            for id in unsup_ids:
                data_list.append(self.unsup_set[id])
            return iter(data_list)
        else:
            data_list=[]
            worker_id = worker_info.id
            if(worker_id<self.sup_size):
                sup_id=random.sample(range(self.sup_len),1)[0]
                data=self.sup_set[sup_id]
            else:
                unsup_id=random.sample(range(self.unsup_len),1)[0]
                data=self.unsup_set[unsup_id]
            data_list.append(data)
            return iter(data_list)
    def __len__(self):
        return self.sup_len+self.unsup_len