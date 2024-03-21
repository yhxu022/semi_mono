import json
import ast
import os
import matplotlib.pyplot as plt
def plot(jsonfile):
    savedir = os.path.dirname(jsonfile)
    batch_unsup_gt_instances_num=[]
    batch_masked_pseudo_instances_num=[]
    batch_cls_unsup_pseudo_instances_num=[]
    batch_regression_unsup_pseudo_instances_num=[]
    loss=[]
    train_iter=[]
    teacher_car_moderate=[]
    student_car_moderate=[]
    step=[]
    lr=[]
    sup_loss_list = []
    unsup_loss_list = []
    consisent_loss_list = []
    # 从JSON文件中读取数据
    with open(jsonfile, 'r') as f:
        for line in f:
            data = dict(item.split(": ") for item in line.strip("\n{}").split(", "))
            new_data={}
            for k in data:
                new_data[ast.literal_eval(k)]=ast.literal_eval(data[k])
            if "batch_unsup_gt_instances_num" in new_data:
                batch_unsup_gt_instances_num.append(new_data["batch_unsup_gt_instances_num"])
            if "batch_masked_pseudo_instances_num" in new_data:
                batch_masked_pseudo_instances_num.append(new_data["batch_masked_pseudo_instances_num"])
            if "batch_cls_unsup_pseudo_instances_num" in new_data:
                batch_cls_unsup_pseudo_instances_num.append(new_data["batch_cls_unsup_pseudo_instances_num"])
            if "batch_regression_unsup_pseudo_instances_num" in new_data:
                batch_regression_unsup_pseudo_instances_num.append(new_data["batch_regression_unsup_pseudo_instances_num"])
            if "loss" in new_data:
                loss.append(new_data["loss"])
                train_iter.append(new_data["iter"])
                sup_loss = 0
                unsup_loss = 0
                for key in new_data.keys():
                    if "sup_loss" in key and "unsup_loss" not in key:
                        sup_loss = sup_loss + new_data[key]
                    if "unsup_loss" in key:
                        unsup_loss = unsup_loss + new_data[key]

                sup_loss_list.append(sup_loss)
                unsup_loss_list.append(unsup_loss)
                if "unsup_consistency_loss" in new_data:
                    consisent_loss_list.append(new_data["unsup_consistency_loss"])
            if "lr" in new_data:
                lr.append(new_data["lr"])
            if "teacher/car_moderate" in new_data:
                teacher_car_moderate.append(new_data["teacher/car_moderate"])
            if "student/car_moderate" in new_data:
                student_car_moderate.append(new_data["student/car_moderate"])
                step.append(new_data["step"])


    plt.figure(1)
    plt.plot(train_iter, batch_unsup_gt_instances_num, label='batch_unsup_gt_instances_num')
    plt.plot(train_iter, batch_masked_pseudo_instances_num, label="batch_masked_pseudo_instances_num")
    plt.plot(train_iter, batch_cls_unsup_pseudo_instances_num, label="batch_cls_unsup_pseudo_instances_num")
    plt.plot(train_iter, batch_regression_unsup_pseudo_instances_num, label="batch_regression_unsup_pseudo_instances_num")
    plt.xlabel('train_iter')
    plt.ylabel('batch_unsup_instances_num')
    plt.legend()
    plt.savefig(os.path.join(savedir,'batch_unsup_gt_instances_num.png'), dpi=1000)
    plt.figure(2)
    plt.plot(train_iter, loss, label='loss')
    plt.plot(train_iter, sup_loss_list, label='sup_loss')
    plt.plot(train_iter, unsup_loss_list, label='unsup_loss')
    plt.plot(train_iter, consisent_loss_list, label='consistent_loss')
    plt.xlabel('train_iter')
    plt.ylabel('loss')   
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss.png'), dpi=1000)
    plt.figure(3)
    plt.plot(step, teacher_car_moderate, label='teacher_car_moderate')
    plt.plot(step, student_car_moderate, label="student_car_moderate")
    plt.xlabel('step')
    plt.ylabel('car_moderate')
    plt.legend()
    plt.savefig(os.path.join(savedir,'car_moderate.png'), dpi=1000)
    plt.figure(4)
    plt.plot(train_iter, lr, label='learning rate')
    plt.xlabel('train_iter')
    plt.ylabel('car_moderate')
    plt.legend()
    plt.savefig(os.path.join(savedir,'learning rate.png'), dpi=1000)
if __name__ == "__main__":
    plot("/data/ipad_3d/monocular/semi_mono/outputs/monodetr_4gpu_2stages_30pc@032120/20240321_202921/vis_data/20240321_202921.json")
