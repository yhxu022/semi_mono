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
    teacher_Car_aos_easy_R40=[]
    teacher_Car_aos_moderate_R40=[]
    teacher_Car_aos_hard_R40=[]
    student_Car_aos_easy_R40=[]
    student_Car_aos_moderate_R40=[]
    student_Car_aos_hard_R40=[]
    teacher_Car_3d_easy_R40=[]
    teacher_Car_3d_moderate_R40=[]
    teacher_Car_3d_hard_R40=[]
    student_Car_3d_easy_R40=[]
    student_Car_3d_moderate_R40=[]
    student_Car_3d_hard_R40=[]
    teacher_Car_bev_easy_R40=[]
    teacher_Car_bev_moderate_R40=[]
    teacher_Car_bev_hard_R40=[]
    student_Car_bev_easy_R40=[]
    student_Car_bev_moderate_R40=[]
    student_Car_bev_hard_R40=[]
    teacher_Car_image_easy_R40=[]
    teacher_Car_image_moderate_R40=[]
    teacher_Car_image_hard_R40=[]
    student_Car_image_easy_R40=[]
    student_Car_image_moderate_R40=[]
    student_Car_image_hard_R40=[]
    step=[]
    lr=[]
    sup_loss_list = []
    unsup_loss_list = []
    consisent_loss_list = []
    sup_loss_ce=[]
    sup_loss_ce_0=[]
    sup_loss_ce_1=[]
    unsup_loss_ce=[]
    unsup_loss_ce_0=[]
    unsup_loss_ce_1=[]
    sup_loss_bbox=[]
    sup_loss_bbox_0=[]
    sup_loss_bbox_1=[]
    unsup_loss_bbox=[]
    unsup_loss_bbox_0=[]
    unsup_loss_bbox_1=[]
    sup_loss_giou=[]
    sup_loss_giou_0=[]
    sup_loss_giou_1=[]
    unsup_loss_giou=[]
    unsup_loss_giou_0=[]
    unsup_loss_giou_1=[]
    sup_loss_depth=[]
    sup_loss_depth_0=[]
    sup_loss_depth_1=[]
    unsup_loss_depth=[]
    unsup_loss_depth_0=[]
    unsup_loss_depth_1=[]
    sup_loss_dim=[]
    sup_loss_dim_0=[]
    sup_loss_dim_1=[]
    unsup_loss_dim=[]
    unsup_loss_dim_0=[]
    unsup_loss_dim_1=[]
    sup_loss_angle=[]
    sup_loss_angle_0=[]
    sup_loss_angle_1=[]
    unsup_loss_angle=[]
    unsup_loss_angle_0=[]
    unsup_loss_angle_1=[]
    sup_loss_center=[]
    sup_loss_center_0=[]
    sup_loss_center_1=[]
    unsup_loss_center=[]
    unsup_loss_center_0=[]
    unsup_loss_center_1=[]
    sup_loss_depth_map=[]
    unsup_loss_depth_map=[]
    teacher_car_moderate=[]
    student_car_moderate=[]
    step_val=[]
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
                sup_loss_ce.append(new_data["sup_loss_ce"])
                sup_loss_ce_0.append(new_data["sup_loss_ce_0"])
                sup_loss_ce_1.append(new_data["sup_loss_ce_1"])
                if "unsup_loss_ce" in  new_data:
                    unsup_loss_ce.append(new_data["unsup_loss_ce"])
                    unsup_loss_ce_0.append(new_data["unsup_loss_ce_0"])
                    unsup_loss_ce_1.append(new_data["unsup_loss_ce_1"])
                sup_loss_bbox.append(new_data["sup_loss_bbox"])
                sup_loss_bbox_0.append(new_data["sup_loss_bbox_0"])
                sup_loss_bbox_1.append(new_data["sup_loss_bbox_1"])
                if "unsup_loss_bbox" in new_data:
                    unsup_loss_bbox.append(new_data["unsup_loss_bbox"])
                    unsup_loss_bbox_0.append(new_data["unsup_loss_bbox"])
                    unsup_loss_bbox_1.append(new_data["unsup_loss_bbox_1"])
                sup_loss_giou.append(new_data["sup_loss_giou"])
                sup_loss_giou_0.append(new_data["sup_loss_giou_0"])
                sup_loss_giou_1.append(new_data["sup_loss_giou_1"])
                if "unsup_loss_giou" in new_data:
                    unsup_loss_giou.append(new_data["unsup_loss_giou"])
                    unsup_loss_giou_0.append(new_data["unsup_loss_giou_0"])
                    unsup_loss_giou_1.append(new_data["unsup_loss_giou_1"])
                sup_loss_depth.append(new_data["sup_loss_depth"])
                sup_loss_depth_0.append(new_data["sup_loss_depth_0"])
                sup_loss_depth_1.append(new_data["sup_loss_depth_1"])
                if "unsup_loss_depth" in new_data:
                    unsup_loss_depth.append(new_data["unsup_loss_depth"])
                    unsup_loss_depth_0.append(new_data["unsup_loss_depth_0"])
                    unsup_loss_depth_1.append(new_data["unsup_loss_depth_1"])
                sup_loss_dim.append(new_data["sup_loss_dim"])
                sup_loss_dim_0.append(new_data["sup_loss_dim_0"])
                sup_loss_dim_1.append(new_data["sup_loss_dim_1"])
                if "unsup_loss_dim" in new_data:
                    unsup_loss_dim.append(new_data["unsup_loss_dim"])
                    unsup_loss_dim_0.append(new_data["unsup_loss_dim_0"])
                    unsup_loss_dim_1.append(new_data["unsup_loss_dim_1"])
                sup_loss_angle.append(new_data["sup_loss_angle"])
                sup_loss_angle_0.append(new_data["sup_loss_angle_0"])
                sup_loss_angle_1.append(new_data["sup_loss_angle_1"])
                if "unsup_loss_angle" in new_data:
                    unsup_loss_angle.append(new_data["unsup_loss_angle"])
                    unsup_loss_angle_0.append(new_data["unsup_loss_angle_0"])
                    unsup_loss_angle_1.append(new_data["unsup_loss_angle_1"])
                sup_loss_center.append(new_data["sup_loss_center"])
                sup_loss_center_0.append(new_data["sup_loss_center_0"])
                sup_loss_center_1.append(new_data["sup_loss_center_1"])
                if "unsup_loss_center" in new_data:
                    unsup_loss_center.append(new_data["unsup_loss_center"])
                    unsup_loss_center_0.append(new_data["unsup_loss_center_0"])
                    unsup_loss_center_1.append(new_data["unsup_loss_center_1"])
                sup_loss_depth_map.append(new_data["sup_loss_depth_map"])
                if "unsup_loss_depth_map" in new_data:
                    unsup_loss_depth_map.append(new_data["unsup_loss_depth_map"])
                sup_loss_list.append(sup_loss)
                unsup_loss_list.append(unsup_loss)
                if "unsup_consistency_loss" in new_data:
                    consisent_loss_list.append(new_data["unsup_consistency_loss"])
            if "lr" in new_data:
                lr.append(new_data["lr"])
            if "teacher/Car_aos_easy_R40" in new_data:
                teacher_Car_aos_easy_R40.append(new_data["teacher/Car_aos_easy_R40"])
                step.append(new_data["step"])
            if "teacher/Car_aos_moderate_R40" in new_data:
                teacher_Car_aos_moderate_R40.append(new_data["teacher/Car_aos_moderate_R40"])
            if "teacher/Car_aos_hard_R40" in new_data:
                teacher_Car_aos_hard_R40.append(new_data["teacher/Car_aos_hard_R40"])
            if "student/Car_aos_easy_R40" in new_data:
                student_Car_aos_easy_R40.append(new_data["student/Car_aos_easy_R40"])
            if "student/Car_aos_moderate_R40" in new_data:
                student_Car_aos_moderate_R40.append(new_data["student/Car_aos_moderate_R40"])   
            if "student/Car_aos_hard_R40" in new_data:
                student_Car_aos_hard_R40.append(new_data["student/Car_aos_hard_R40"])
            if "teacher/Car_3d_easy_R40" in new_data:
                teacher_Car_3d_easy_R40.append(new_data["teacher/Car_3d_easy_R40"])
            if "teacher/Car_3d_moderate_R40" in new_data:
                teacher_Car_3d_moderate_R40.append(new_data["teacher/Car_3d_moderate_R40"])
            if "teacher/Car_3d_hard_R40" in new_data:
                teacher_Car_3d_hard_R40.append(new_data["teacher/Car_3d_hard_R40"])
            if "student/Car_3d_easy_R40" in new_data:
                student_Car_3d_easy_R40.append(new_data["student/Car_3d_easy_R40"])
            if "student/Car_3d_moderate_R40" in new_data:
                student_Car_3d_moderate_R40.append(new_data["student/Car_3d_moderate_R40"])
            if "student/Car_3d_hard_R40" in new_data:
                student_Car_3d_hard_R40.append(new_data["student/Car_3d_hard_R40"])
            if "teacher/Car_bev_easy_R40" in new_data:
                teacher_Car_bev_easy_R40.append(new_data["teacher/Car_bev_easy_R40"])
            if "teacher/Car_bev_moderate_R40" in new_data:
                teacher_Car_bev_moderate_R40.append(new_data["teacher/Car_bev_moderate_R40"])
            if "teacher/Car_bev_hard_R40" in new_data:
                teacher_Car_bev_hard_R40.append(new_data["teacher/Car_bev_hard_R40"])
            if "student/Car_bev_easy_R40" in new_data:
                student_Car_bev_easy_R40.append(new_data["student/Car_bev_easy_R40"])
            if "student/Car_bev_moderate_R40" in new_data:
                student_Car_bev_moderate_R40.append(new_data["student/Car_bev_moderate_R40"])
            if "student/Car_bev_hard_R40" in new_data:
                student_Car_bev_hard_R40.append(new_data["student/Car_bev_hard_R40"])
            if "teacher/Car_image_easy_R40" in new_data:
                teacher_Car_image_easy_R40.append(new_data["teacher/Car_image_easy_R40"])
            if "teacher/Car_image_moderate_R40" in new_data:
                teacher_Car_image_moderate_R40.append(new_data["teacher/Car_image_moderate_R40"])
            if "teacher/Car_image_hard_R40" in new_data:
                teacher_Car_image_hard_R40.append(new_data["teacher/Car_image_hard_R40"])
            if "student/Car_image_easy_R40" in new_data:
                student_Car_image_easy_R40.append(new_data["student/Car_image_easy_R40"])
            if "student/Car_image_moderate_R40" in new_data:
                student_Car_image_moderate_R40.append(new_data["student/Car_image_moderate_R40"])
            if "student/Car_image_hard_R40" in new_data:
                student_Car_image_hard_R40.append(new_data["student/Car_image_hard_R40"])
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
    plt.plot(step, teacher_Car_aos_easy_R40, label='teacher_Car_aos_easy_R40')
    plt.plot(step, teacher_Car_aos_moderate_R40, label='teacher_Car_aos_moderate_R40')
    plt.plot(step, teacher_Car_aos_hard_R40, label='teacher_Car_aos_hard_R40')
    plt.plot(step, student_Car_aos_easy_R40, label="student_Car_aos_easy_R40")
    plt.plot(step, student_Car_aos_moderate_R40, label="student_Car_aos_moderate_R40")
    plt.plot(step, student_Car_aos_hard_R40, label="student_Car_aos_hard_R40")
    plt.xlabel('step')
    plt.ylabel('Car_aos')
    plt.legend()
    plt.savefig(os.path.join(savedir,'Car_aos.png'), dpi=1000)
    plt.figure(4)
    plt.plot(train_iter, lr, label='learning rate')
    plt.xlabel('train_iter')
    plt.ylabel('car_moderate')
    plt.legend()
    plt.savefig(os.path.join(savedir,'learning rate.png'), dpi=1000)
    plt.figure(5)
    plt.plot(step, teacher_Car_3d_easy_R40, label='teacher_Car_3d_easy_R40')
    plt.plot(step, teacher_Car_3d_moderate_R40, label='teacher_Car_3d_moderate_R40')
    plt.plot(step, teacher_Car_3d_hard_R40, label='teacher_Car_3d_hard_R40')
    plt.plot(step, student_Car_3d_easy_R40, label="student_Car_3d_easy_R40")
    plt.plot(step, student_Car_3d_moderate_R40, label="student_Car_3d_moderate_R40")
    plt.plot(step, student_Car_3d_hard_R40, label="student_Car_3d_hard_R40")
    plt.xlabel('step')
    plt.ylabel('Car_3d')
    plt.legend()
    plt.savefig(os.path.join(savedir,'Car_3d.png'), dpi=1000)
    plt.figure(6)
    plt.plot(step, teacher_Car_bev_easy_R40, label='teacher_Car_bev_easy_R40')
    plt.plot(step, teacher_Car_bev_moderate_R40, label='teacher_Car_bev_moderate_R40')
    plt.plot(step, teacher_Car_bev_hard_R40, label='teacher_Car_bev_hard_R40')
    plt.plot(step, student_Car_bev_easy_R40, label="student_Car_bev_easy_R40")
    plt.plot(step, student_Car_bev_moderate_R40, label="student_Car_bev_moderate_R40")
    plt.plot(step, student_Car_bev_hard_R40, label="student_Car_bev_hard_R40")
    plt.xlabel('step')
    plt.ylabel('Car_bev')
    plt.legend()
    plt.savefig(os.path.join(savedir,'Car_bev.png'), dpi=1000)
    plt.figure(7)
    plt.plot(step, teacher_Car_image_easy_R40, label='teacher_Car_image_easy_R40')
    plt.plot(step, teacher_Car_image_moderate_R40, label='teacher_Car_image_moderate_R40')
    plt.plot(step, teacher_Car_image_hard_R40, label='teacher_Car_image_hard_R40')
    plt.plot(step, student_Car_image_easy_R40, label="student_Car_image_easy_R40")
    plt.plot(step, student_Car_image_moderate_R40, label="student_Car_image_moderate_R40")
    plt.plot(step, student_Car_image_hard_R40, label="student_Car_image_hard_R40")
    plt.xlabel('step')
    plt.ylabel('Car_image')
    plt.legend()
    plt.savefig(os.path.join(savedir,'Car_image.png'), dpi=1000)  
    plt.figure(8)    
    plt.plot(train_iter, sup_loss_ce, label='sup_loss_ce')
    plt.plot(train_iter, sup_loss_ce_0, label='sup_loss_ce_0')
    plt.plot(train_iter, sup_loss_ce_1, label='sup_loss_ce_1')
    if len(unsup_loss_ce)>0:
        plt.plot(train_iter, unsup_loss_ce, label='unsup_loss_ce')
        plt.plot(train_iter, unsup_loss_ce_0, label='unsup_loss_ce_0')
        plt.plot(train_iter, unsup_loss_ce_1, label='unsup_loss_ce_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_ce')   
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_ce.png'), dpi=1000)
    plt.figure(9)
    plt.plot(train_iter, sup_loss_giou, label='sup_loss_giou')
    plt.plot(train_iter, sup_loss_giou_0, label='sup_loss_giou_0')
    plt.plot(train_iter, sup_loss_giou_1, label='sup_loss_giou_1')
    if len(unsup_loss_giou)>0:
        plt.plot(train_iter, unsup_loss_giou, label='unsup_loss_giou')
        plt.plot(train_iter, unsup_loss_giou_0, label='unsup_loss_giou_0')
        plt.plot(train_iter, unsup_loss_giou_1, label='unsup_loss_giou_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_giou')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_giou.png'), dpi=1000)
    plt.figure(10)
    plt.plot(train_iter, sup_loss_bbox, label='sup_loss_bbox')
    plt.plot(train_iter, sup_loss_bbox_0, label='sup_loss_bbox_0')
    plt.plot(train_iter, sup_loss_bbox_1, label='sup_loss_bbox_1')
    if len(unsup_loss_bbox)>0:
        plt.plot(train_iter, unsup_loss_bbox, label='unsup_loss_bbox')
        plt.plot(train_iter, unsup_loss_bbox_0, label='unsup_loss_bbox_0')
        plt.plot(train_iter, unsup_loss_bbox_1, label='unsup_loss_bbox_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_bbox')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_bbox.png'), dpi=1000)
    plt.figure(11)
    plt.plot(train_iter, sup_loss_depth, label='sup_loss_depth')
    plt.plot(train_iter, sup_loss_depth_0, label='sup_loss_depth_0')
    plt.plot(train_iter, sup_loss_depth_1, label='sup_loss_depth_1')
    if len(unsup_loss_depth)>0:
        plt.plot(train_iter, unsup_loss_depth, label='unsup_loss_depth')
        plt.plot(train_iter, unsup_loss_depth_0, label='unsup_loss_depth_0')
        plt.plot(train_iter, unsup_loss_depth_1, label='unsup_loss_depth_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_depth')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_depth.png'), dpi=1000)
    plt.figure(12)
    plt.plot(train_iter, sup_loss_dim, label='sup_loss_dim')
    plt.plot(train_iter, sup_loss_dim_0, label='sup_loss_dim_0')
    plt.plot(train_iter, sup_loss_dim_1, label='sup_loss_dim_1')
    if len(unsup_loss_dim)>0:
        plt.plot(train_iter, unsup_loss_dim, label='unsup_loss_dim')
        plt.plot(train_iter, unsup_loss_dim_0, label='unsup_loss_dim_0')
        plt.plot(train_iter, unsup_loss_dim_1, label='unsup_loss_dim_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_dim')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_dim.png'), dpi=1000)
    plt.figure(13)
    plt.plot(train_iter, sup_loss_angle, label='sup_loss_angle')
    plt.plot(train_iter, sup_loss_angle_0, label='sup_loss_angle_0')
    plt.plot(train_iter, sup_loss_angle_1, label='sup_loss_angle_1')
    if len(unsup_loss_angle)>0:
        plt.plot(train_iter, unsup_loss_angle, label='unsup_loss_angle')
        plt.plot(train_iter, unsup_loss_angle_0, label='unsup_loss_angle_0')
        plt.plot(train_iter, unsup_loss_angle_1, label='unsup_loss_angle_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_angle')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_angle.png'), dpi=1000)
    plt.figure(14)
    plt.plot(train_iter, sup_loss_center, label='sup_loss_center')
    plt.plot(train_iter, sup_loss_center_0, label='sup_loss_center_0')
    plt.plot(train_iter, sup_loss_center_1, label='sup_loss_center_1')
    if len(unsup_loss_center)>0:
        plt.plot(train_iter, unsup_loss_center, label='unsup_loss_center')
        plt.plot(train_iter, unsup_loss_center_0, label='unsup_loss_center_0')
        plt.plot(train_iter, unsup_loss_center_1, label='unsup_loss_center_1')
    plt.xlabel('train_iter')
    plt.ylabel('loss_center')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_center.png'), dpi=1000)
    plt.figure(15)
    plt.plot(train_iter, sup_loss_depth_map, label='sup_loss_depth_map')
    if len(unsup_loss_depth_map)>0:
        plt.plot(train_iter, unsup_loss_depth_map, label='unsup_loss_depth_map')
    plt.xlabel('train_iter')
    plt.ylabel('loss_depth_map')
    plt.legend()
    plt.savefig(os.path.join(savedir,'loss_depth_map.png'), dpi=1000)
    plt.figure(16)
    plt.plot(step, teacher_Car_3d_moderate_R40, label='teacher_Car_3d_moderate_R40')
    plt.plot(step, student_Car_3d_moderate_R40, label='student_Car_3d_moderate_R40')
    plt.xlabel('step')
    plt.ylabel('Car_3d_moderate_R40')
    plt.legend()
    plt.savefig(os.path.join(savedir, 'Car_3d_moderate_R40.png'), dpi=1000)
if __name__ == "__main__":
    plot("/data/ipad_3d/monocular/semi_mono/outputs/monodetr_4gpu_2stages_30pc@032711/20240327_114918/vis_data/20240327_114918.json")
