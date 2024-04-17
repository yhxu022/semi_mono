import torch

def bbox_iou(bbox1, bbox2):
    """
    计算两组矩形框之间的IoU矩阵。
    参数:
    bbox1: [N, 4]，每一行为[x, y, dx, dy]
    bbox2: [M, 4]，每一行为[x, y, dx, dy]
    返回:
    iou_2d: [N, M]，IoU值矩阵
    """
    # 转换bbox格式从[x, y, dx, dy]到[x1, y1, x2, y2]
    bbox1[:, 2:] += bbox1[:, :2]
    bbox2[:, 2:] += bbox2[:, :2]

    # 计算N和M边界框的交点坐标
    x1 = torch.max(bbox1[:, None, 0], bbox2[:, 0])  # [N, M]
    y1 = torch.max(bbox1[:, None, 1], bbox2[:, 1])  # [N, M]
    x2 = torch.min(bbox1[:, None, 2], bbox2[:, 2])  # [N, M]
    y2 = torch.min(bbox1[:, None, 3], bbox2[:, 3])  # [N, M]

    # 计算交集区域的宽度和高度
    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)

    # 计算交集和并集面积
    intersection_area = inter_width * inter_height
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

    union_area = area1[:, None] + area2 - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou