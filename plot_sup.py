import json
import ast
import os
import matplotlib.pyplot as plt
def plot(jsonfile,savedir):
    savedir = os.path.dirname(jsonfile)
    batch_unsup_gt_instances_num=[]
    batch_unsup_pseudo_instances_num=[]
    loss=[]
    train_iter=[]
    car_moderate=[]
    car_moderate=[]
    student_car_moderate=[]
    step=[]
    Car_aos_easy_R40=[]
    Car_aos_moderate_R40=[]
    Car_aos_hard_R40=[]
    Car_3d_easy_R40=[]
    Car_3d_moderate_R40=[]
    Car_3d_hard_R40=[]
    Car_bev_easy_R40=[]
    Car_bev_moderate_R40=[]
    Car_bev_hard_R40=[]
    Car_image_easy_R40=[]
    Car_image_moderate_R40=[]
    Car_image_hard_R40=[]
    # 从JSON文件中读取数据
    with open(jsonfile, 'r') as f:
        for line in f:
            data = dict(item.split(": ") for item in line.strip("\n{}").split(", "))
            new_data={}
            for k in data:
                new_data[ast.literal_eval(k)]=ast.literal_eval(data[k])
            if "batch_unsup_gt_instances_num" in new_data:
                batch_unsup_gt_instances_num.append(new_data["batch_unsup_gt_instances_num"])
            if "batch_unsup_pseudo_instances_num" in new_data:
                batch_unsup_pseudo_instances_num.append(new_data["batch_unsup_pseudo_instances_num"])
            if "loss" in new_data:
                loss.append(new_data["loss"])
                train_iter.append(new_data["iter"])
            # if "car_moderate" in new_data:
            #     car_moderate.append(new_data["car_moderate"])
            #     step.append(new_data["step"])
            # if "car_moderate" in new_data:
            #     car_moderate.append(new_data["car_moderate"])
            if "Car_aos_easy_R40" in new_data:
                Car_aos_easy_R40.append(new_data["Car_aos_easy_R40"])

            if "Car_aos_moderate_R40" in new_data:
                Car_aos_moderate_R40.append(new_data["Car_aos_moderate_R40"])
            if "Car_aos_hard_R40" in new_data:
                Car_aos_hard_R40.append(new_data["Car_aos_hard_R40"])
            if "Car_3d_easy_R40" in new_data:
                Car_3d_easy_R40.append(new_data["Car_3d_easy_R40"])
            if "Car_3d_moderate_R40" in new_data:
                Car_3d_moderate_R40.append(new_data["Car_3d_moderate_R40"])
                step.append(new_data["step"])
            if "Car_3d_hard_R40" in new_data:
                Car_3d_hard_R40.append(new_data["Car_3d_hard_R40"])
            if "Car_bev_easy_R40" in new_data:
                Car_bev_easy_R40.append(new_data["Car_bev_easy_R40"])
            if "Car_bev_moderate_R40" in new_data:
                Car_bev_moderate_R40.append(new_data["Car_bev_moderate_R40"])
            if "Car_bev_hard_R40" in new_data:
                Car_bev_hard_R40.append(new_data["Car_bev_hard_R40"])
            if "Car_image_easy_R40" in new_data:
                Car_image_easy_R40.append(new_data["Car_image_easy_R40"])
            if "Car_image_moderate_R40" in new_data:
                Car_image_moderate_R40.append(new_data["Car_image_moderate_R40"])
            if "Car_image_hard_R40" in new_data:
                Car_image_hard_R40.append(new_data["Car_image_hard_R40"])
    if(len(Car_3d_moderate_R40)==0):
        plt.figure(1)
        plt.plot(train_iter, batch_unsup_gt_instances_num, label='batch_unsup_gt_instances_num')
        plt.plot(train_iter, batch_unsup_pseudo_instances_num, label="batch_unsup_pseudo_instances_num")
        plt.xlabel('train_iter')
        plt.ylabel('batch_unsup_instances_num')
        plt.legend()
        plt.savefig(os.path.join(savedir,'batch_unsup_gt_instances_num.png'), dpi=1000)
        plt.figure(2)
        plt.plot(train_iter, loss, label='loss')
        plt.xlabel('train_iter')
        plt.ylabel('loss')   
        plt.legend()
        plt.savefig(os.path.join(savedir,'loss.png'), dpi=1000)
        plt.figure(3)
        plt.plot(step, car_moderate, label='car_moderate')
        plt.plot(step, student_car_moderate, label="student_car_moderate")
        plt.xlabel('step')
        plt.ylabel('car_moderate')
        plt.legend()
        plt.savefig(os.path.join(savedir,'car_moderate.png'), dpi=1000)
    else:
        plt.figure(1)
        plt.plot(train_iter, loss, label='loss')
        plt.xlabel('train_iter')
        plt.ylabel('loss')   
        plt.legend()
        plt.savefig(os.path.join(savedir,'sup_loss.png'), dpi=1000)
        plt.figure(2)
        plt.plot(step, Car_3d_moderate_R40, label='car_moderate')
        plt.xlabel('step')
        plt.ylabel('car_moderate')
        plt.legend()
        plt.savefig(os.path.join(savedir,'car_moderate.png'), dpi=1000)
if __name__ == "__main__":
    plot("/home/xyh/MonoDETR_semi_baseline_33/outputs/monodetr_100pc+eigenclean@032720_24/20240327_202410/vis_data/20240327_202410.json",

         None)