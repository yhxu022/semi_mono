from mmengine.evaluator import BaseMetric
import os
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
class KITTI_METRIC(BaseMetric):
    def __init__(self,
                  dataloader,
                  logger,
                  cfg
                  ):
        super().__init__()
        self.output_dir = os.path.join('./' + cfg["trainer"]['save_path'], cfg["model_name"])
        self.dataloader = dataloader
        self.logger=logger
        self.class_name = dataloader["dataset"].class_name
        self.max_objs = dataloader["dataset"].max_objs    # max objects per images, defined in dataset


    def process(self, data_batch, data_samples):
        dets, targets = data_samples
        self.save_results(dets)
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': 0,
            'correct': 0,
        })

    def compute_metrics(self, results):
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(car_moderate=self.kitti_evaluate())
    
    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def kitti_evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result =self.dataloader["dataset"].eval(results_dir=results_dir, logger=self.logger)
        return result