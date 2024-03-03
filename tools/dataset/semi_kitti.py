import argparse
import numpy as np
import json
import os
import pickle
import mmengine
from mmengine.utils import mkdir_or_exist, track_parallel_progress

def save_txt(out_dir, name, images):
    mkdir_or_exist(out_dir)
    for i in range(len(images) - 1):
        images[i] = images[i] + "\n"
    # 使用open()函数和writelines()方法将文本逐行写入txt文件
    with open(f'{out_dir}/{name}.txt', "w") as file:
        # 将文本逐行写入文件
        file.writelines(images)

def _save_anno(save_name,data_info, data_list, save_dir, key):
    """Save annotation information for semi-supervised learning."""
    """key: labeled or unlabeled """
    save_path = os.path.join(save_dir, f"{save_name}.pkl")

    new_data = {}
    new_data['metainfo'] = data_info
    new_data['data_list'] = data_list
    # if key is "labeled":
    #     new_data['metainfo'] = data_info
    #     new_data['data_list'] = data_list
    # elif key is "unlabeled":
    #     new_data['metainfo'] = data_info
    #     for idx, single_data in enumerate(data_list):
    #         new_single_data = single_data
    #         new_single_data[]
    #         data_list[idx] = new_single_data
    #
    #     new_data['data_list'] = data_list
    # else:
    #     raise ValueError(f"key is not \"labeled\" or \"unlabeled\", which is {key}")
    # with open(save_path, 'wb') as f:
    mmengine.dump(new_data, save_path)
    print(f"Annotation saved at {save_path}")

def prepare_kitti_data(seed=444, percent=10.0, data_dir='data', ktii_dir="KITTI", seed_offset=0):
    """Prepare KITTI dataset for Semi-supervised learning
    Args:
      seed: random seed for dataset split.
      percent: percentage of labeled dataset.
      data_dir: root directory of KITTI dataset.
      seed_offset: offset to be added to the seed for randomization.
    """
    kitti_data_dir = os.path.join(data_dir, ktii_dir)
    np.random.seed(seed + seed_offset)
    # Load KITTI meta information
    info_train = pickle.load(open(os.path.join(kitti_data_dir, 'kitti_infos_train.pkl'), 'rb'))
    data_info = info_train['metainfo']
    data_train = info_train['data_list']

    # Splitting data into labeled and unlabeled subsets
    num_labeled = int((percent / 100.0) * len(data_train))
    labeled_indices = np.random.choice(len(data_train), size=num_labeled, replace=False)

    labeled_data = [data_train[idx] for idx in labeled_indices]
    unlabeled_data = [info for idx, info in enumerate(data_train) if idx not in labeled_indices]

    save_path = "{}/{}".format(kitti_data_dir, "semi_supervised_pkl")
    os.makedirs(save_path, exist_ok=True)

    # Save labeled and unlabeled data annotations
    labeled_save_name = "kitti_infos_train.{seed}@{tot}".format(
        seed=seed, tot=int(percent)
    )
    _save_anno(labeled_save_name, data_info, labeled_data, save_path, "labeled")
    unlabeled_save_name = "kitti_infos_train.{seed}@{tot}-unlabeled".format(
        seed=seed, tot=int(percent)
    )
    _save_anno(unlabeled_save_name, data_info, unlabeled_data, save_path, "unlabeled")

    file = os.path.join(kitti_data_dir, "ImageSets", 'train.txt')
    # 使用open()函数加载txt文件
    with open(file, "r") as file:
        imgs = file.readlines()
    for i in range(len(imgs)):
        imgs[i] = imgs[i].strip("\n")

    labeled_inds = set(labeled_indices)
    labeled_images, unlabeled_images = [], []

    for i in range(len(imgs)):
        if i in labeled_inds:
            labeled_images.append(imgs[i])
        else:
            unlabeled_images.append(imgs[i])

    # save labeled and unlabeled
    labeled_name = "train.{seed}@{tot}".format(
        seed=seed, tot=int(percent)
    )
    unlabeled_name = "train.{seed}@{tot}-unlabeled".format(
        seed=seed, tot=int(percent)
    )

    txt_dir = "{}/{}/{}".format(data_dir, ktii_dir, "semi_supervised_ImageSets")
    os.makedirs(save_path, exist_ok=True)
    save_txt(txt_dir, labeled_name, labeled_images)
    save_txt(txt_dir, unlabeled_name, unlabeled_images)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--percent", type=float, default=10)
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    print(args)
    prepare_kitti_data(seed=args.seed, percent=args.percent, data_dir=args.data_dir, ktii_dir="KITTI", seed_offset=args.seed_offset)
