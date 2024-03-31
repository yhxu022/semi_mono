import numpy as np
from torch.utils.data import DataLoader
#一般训练时使用
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
#测试Val图片做弱增强时使用
# from lib.datasets.kitti.kitti_dataset_aug import KITTI_Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import RandomSampler
from tools.semi_sampler import Semi_Sampler


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=16):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        if cfg['train_split'] == 'semi':
            if cfg['percent'] != 100:
                labeled_dataset = KITTI_Dataset(split=cfg['train_split'] + "_labeled", cfg=cfg)
            else:
                labeled_dataset = KITTI_Dataset(split='train', cfg=cfg)
            unlabeled_dataset = KITTI_Dataset(split=cfg['train_split'] + "_unlabeled", cfg=cfg)
            train_set = ConcatDataset([labeled_dataset, unlabeled_dataset])
            sampler = Semi_Sampler(len(labeled_dataset), len(unlabeled_dataset), cfg['batch_size'], cfg['sup_size'])
        elif cfg['train_split'] == 'semi_eigen_clean':
            if 'fold' in cfg and 'percent' in cfg:
                if cfg['percent'] != 100:
                    labeled_dataset = KITTI_Dataset(split='semi' + "_labeled", cfg=cfg)
                else:
                    labeled_dataset = KITTI_Dataset(split='train', cfg=cfg)
            else:
                labeled_dataset = KITTI_Dataset(split='train', cfg=cfg)
            unlabeled_dataset = KITTI_Dataset(split='eigen_clean', cfg=cfg)
            train_set = ConcatDataset([labeled_dataset, unlabeled_dataset])
            sampler = Semi_Sampler(len(labeled_dataset), len(unlabeled_dataset), cfg['batch_size'], cfg['sup_size'])
        elif cfg['train_split'] == 'semi_raw_mix':
            if 'fold' in cfg and 'percent' in cfg:
                if cfg['percent'] != 100:
                    labeled_dataset = KITTI_Dataset(split='semi' + "_labeled", cfg=cfg)
                else:
                    labeled_dataset = KITTI_Dataset(split='train', cfg=cfg)
            else:
                labeled_dataset = KITTI_Dataset(split='train', cfg=cfg)
            unlabeled_dataset = KITTI_Dataset(split='raw_mix', cfg=cfg)
            train_set = ConcatDataset([labeled_dataset, unlabeled_dataset])
            sampler = Semi_Sampler(len(labeled_dataset), len(unlabeled_dataset), cfg['batch_size'], cfg['sup_size'])
        else:
            print(cfg['train_split'])
            unlabeled_dataset = None
            if cfg['train_split'] == '100pc+eigen_clean':
                train_set_kitti3D = KITTI_Dataset(split='train', cfg=cfg)
                train_set_eigen_clean = KITTI_Dataset(split='eigen_clean_sup', cfg=cfg)
                train_set = ConcatDataset([train_set_kitti3D, train_set_eigen_clean])
                sampler = RandomSampler(train_set, replacement=True, num_samples=800000)
            else:
                train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
                sampler = RandomSampler(train_set, replacement=True, num_samples=800000)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])
    # test_loader = DataLoader(dataset=test_set,
    #                          batch_size=4,
    #                          num_workers=8,
    #                          worker_init_fn=my_worker_init_fn,
    #                          shuffle=False,
    #                          pin_memory=True,
    #                          drop_last=False,
    #                          persistent_workers=True)
    test_dataloader = dict(
        batch_size=cfg.get("val_batch_size",4),
        sampler=dict(
            type='DefaultSampler',
            shuffle=False),
        dataset=test_set,
        pin_memory=True,
        num_workers=cfg.get("val_num_workers",8),
        collate_fn=dict(type='default_collate'),
        persistent_workers=True
    )
    if cfg.get('cls_GT', False) is True:
        return train_set, test_dataloader, sampler, unlabeled_dataset
    else:
        return train_set, test_dataloader, sampler
