import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from tools.Semi_KITTI import Semi_KITTI
from torch.utils.data import ConcatDataset
from tools.semi_sampler import Semi_Sampler
# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
def build_dataloader(cfg, workers=16):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        labeled_dataset = KITTI_Dataset(split=cfg['train_split']+"_labeled", cfg=cfg)
        unlabeled_dataset = KITTI_Dataset(split=cfg['train_split']+"_unlabeled", cfg=cfg)
        train_set=ConcatDataset([labeled_dataset,unlabeled_dataset])
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])
    test_loader = DataLoader(dataset=test_set,
                             batch_size=4,
                             num_workers=8,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             persistent_workers=True)
    test_dataloader=dict(
        batch_size=4,
        sampler=dict(
            type='DefaultSampler',
            shuffle=False),
        dataset=test_set,
        pin_memory=True,
        num_workers=8,
        collate_fn=dict(type='default_collate'),
        persistent_workers=True
    )
    return train_set, test_dataloader, Semi_Sampler(len(labeled_dataset),len(unlabeled_dataset),cfg['batch_size'],cfg['sup_size'])