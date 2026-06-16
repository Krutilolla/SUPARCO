from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

def find_corners_coordinates(cx, cy, w, h, theta, use_radians = False):
    """
    The function takes the arguments of center x coordinate, center y coordinate, width, height and inclination 
    angle from the horizontal direction and returns the four coordinates of the rectangle.
    The formula used is shown below
    corners = [cos(θ), -sin(θ)] [-w/2, w/2, w/2, -w/2]     [cx, cx, cx, cx]
              [sin(θ),  cos(θ)] [-h/2, -h/2, h/2, h/2]  +  [cy, cy, cy, cy]
                                       

    """
    # Define the initial rectangle
    rectangle = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])

    # Rotate and translate the rectangle
    rotated_rect = rotate(rectangle, theta, use_radians=use_radians)
    obb = translate(rotated_rect, cx, cy)
    return obb


def find_rotated_iou(box1, box2, use_radians = False):
    """
    Takes centre coordinates, width, height and theta for the 2 rectangles and return IoU
    This is implemented through https://medium.com/mindkosh/calculating-iou-between-oriented-bounding-boxes-c39f72602cac 

    Arguments:
    box1, box2: list of [x coordinate of centre, y coordinate of centre, width, height, angle of inclination] 
    for box1 and box2
    use_radians: if the angles given are in radians

    Returns:
    IoU = Area of Intersection/Area of Overlap
    """

    # cx1, cy1, w1, h1, theta1 = box1
    # cx2, cy2, w2, h2, theta2 = box2


    # Define the OBBs
    # obb1 = find_corners_coordinates(cx1, cy1, w1, h1, theta1, use_radians)
    # obb2 = find_corners_coordinates(cx2, cy2, w2, h2, theta2, use_radians)
    # print(obb1)
    # print(obb2)

    box1 = box1 + [box1[0]]
    box2 = box2 + [box2[0]]


    obb1=Polygon(box1)
    obb2=Polygon(box2)
    print(obb1)
    print(obb2)
    

    # Calculate the intersection area
    intersection_area = obb1.intersection(obb2).area

    # Calculate the union area
    union_area = obb1.area + obb2.area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def evaluate_grounding_score(gt_boxes, pred_boxes, alpha=0.693, iou_threshold=0.0):
    """
    Finds the Grounding score between set of ground truth bounding boxes and predicted bounding boxes for a single image.
    It uses the Hungarian algorithm for the Maximum Weight Bipartite matching used to match the best possible ground truth bounding box and 
    the predicted bounding box

    Arguments:
    gt_boxes, pred_boxes: lists/arrays of ground truth and predicted boxes, single bounding box - [xc,yc,w,h,theta]
    alpha: weighting factor for CP (default ln(2) so 1 count diff halves CP)
    iou_threshold: minimum IoU to consider a match valid (0.0 means all pairs allowed)

    Returns: dict with score, CP i.e Count Penalty, MeanIoU, matched_pairs (list of (gt_idx, pred_idx, iou))
    """

    N_ref = len(gt_boxes)
    N_pred = len(pred_boxes)

    if N_ref == 0 and N_pred == 0:
        return {"score": 1.0, "CP": 1.0, "MeanIoU": 1.0, "matches": []}
    
    if N_ref == 0:
        cp = np.exp(-alpha * N_pred)
        return {"score": 0.0, "CP": cp, "MeanIoU": 0.0, "matches": []}
    
    if N_pred == 0:
        cp = np.exp(-alpha * N_ref)
        return {"score": 0.0, "CP": cp, "MeanIoU": 0.0, "matches": []}
    
    # Build IoU matrix
    pair_wise_iou = np.zeros((N_ref, N_pred), dtype=float)
    for i in range(N_ref):
        for j in range(N_pred):
            pair_wise_iou[i, j] = find_rotated_iou(gt_boxes[i], pred_boxes[j])
    
    matches = []

    
    cost = pair_wise_iou.copy()

    # Maximum weight matching by Hungarian algorithm
    row_idx, col_idx = linear_sum_assignment(cost, maximize = True)

    for r, c in zip(row_idx, col_idx):
        if pair_wise_iou[r, c] >= iou_threshold:
            matches.append((int(r), int(c), float(pair_wise_iou[r, c])))
        else:
            # below threshold => treat as unmatched (IoU=0)
            pass
    
    matched_iou_sum = sum(m for (_, _, m) in matches)
    MeanIoU = matched_iou_sum / max(N_ref, N_pred)   # unmatched boxes counted as IoU=0, the average is being done by big number, this can be relaxed by averaging it by min
    CP = float(np.exp(-alpha * abs(N_pred - N_ref)))
    score = CP * MeanIoU

    return {
        "score": float(score),
        "CP": float(CP),
        "MeanIoU": float(MeanIoU),
        "matches": matches,   # list of tuples (gt_idx, pred_idx, iou)
        "N_ref": N_ref,
        "N_pred": N_pred
    }

# gt_boxes = [[(0,0),(0,2),(2,2),(2,0)], [(0,0),(0,4),(4,4),(4,0)]] # each bounding box is [xc, yc, w, h, theta]
# pred_boxes = [[(0,0),(0,2),(2,2),(2,0)]]
# print(evaluate_grounding_score(gt_boxes, pred_boxes))



