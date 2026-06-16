from system_prompts.yolo_filter_rules import *
import cv2


# ============================================================
# COLOR DETECTION FUNCTIONS
# ============================================================
def corners_to_polygon(corners):
    """Convert 4 corner points to Shapely polygon."""
    try:
        poly = Polygon(corners)
        if not poly.is_valid:
            poly = make_valid(poly)
        return poly
    except:
        return None
        
def get_color_ratio(image_bgr, corners, color_name, threshold=0.10):
    """Calculate the ratio of pixels matching a specific color in an OBB region."""
    if color_name not in COLOR_HSV_RANGES:
        return 0.0
    
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    
    x_min = max(0, int(np.floor(x_coords.min())))
    x_max = min(image_bgr.shape[1], int(np.ceil(x_coords.max())))
    y_min = max(0, int(np.floor(y_coords.min())))
    y_max = min(image_bgr.shape[0], int(np.ceil(y_coords.max())))
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    roi_bgr = image_bgr[y_min:y_max, x_min:x_max]
    if roi_bgr.size == 0:
        return 0.0
    
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    
    shifted_corners = corners - np.array([x_min, y_min])
    obb_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    cv2.fillPoly(obb_mask, [shifted_corners.astype(np.int32)], 255)
    
    color_ranges = COLOR_HSV_RANGES[color_name]
    color_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for range_def in color_ranges:
        mask = cv2.inRange(roi_hsv, range_def["lower"], range_def["upper"])
        color_mask = cv2.bitwise_or(color_mask, mask)
    
    final_mask = cv2.bitwise_and(color_mask, obb_mask)
    
    color_pixels = cv2.countNonZero(final_mask)
    total_pixels = cv2.countNonZero(obb_mask)
    
    if total_pixels == 0:
        return 0.0
    
    return color_pixels / total_pixels


def filter_by_color(predictions, color_name, image_bgr, threshold=0.10):
    """Filter predictions to only include objects with a specific color."""
    if color_name not in COLOR_HSV_RANGES:
        # print(f"⚠️ Unknown color: '{color_name}'. Supported: {sorted(BASE_COLORS)}")
        return predictions
    
    filtered = []
    for pred in predictions:
        ratio = get_color_ratio(image_bgr, pred['corners'], color_name, threshold)
        pred['color_ratio'] = ratio
        if ratio >= threshold:
            filtered.append(pred)
    
    filtered.sort(key=lambda x: x.get('color_ratio', 0), reverse=True)
    return filtered


# ============================================================
# SPATIAL/SIZE/ORDINAL FILTERS
# ============================================================
def get_box_center(corners):
    """Get center of OBB."""
    return corners.mean(axis=0)


def get_box_area(corners):
    """Get area of OBB polygon."""
    poly = corners_to_polygon(corners)
    return poly.area if poly else 0


def filter_by_spatial_region(predictions, region, img_w, img_h):
    """Filter predictions by spatial region."""
    if region not in SPATIAL_REGIONS:
        return predictions
    
    bounds = SPATIAL_REGIONS[region]
    x_min, x_max = bounds["x"][0] * img_w, bounds["x"][1] * img_w
    y_min, y_max = bounds["y"][0] * img_h, bounds["y"][1] * img_h
    
    filtered = []
    for pred in predictions:
        cx, cy = get_box_center(pred['corners'])
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            filtered.append(pred)
    
    return filtered

def get_obb_dimensions(corners):
    """Get length (longer side) and width (shorter side) of OBB."""
    # Calculate the 4 edge lengths
    edges = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        edges.append(edge_len)
    # OBB has 2 pairs of equal edges
    side1 = (edges[0] + edges[2]) / 2
    side2 = (edges[1] + edges[3]) / 2
    length = max(side1, side2)  # longer side
    width = min(side1, side2)   # shorter side
    return length, width

def filter_by_size(predictions, size_filter):
    """Filter/sort predictions by size (area, length, or width)."""
    if not predictions:
        return predictions
    
    # Calculate dimensions for all predictions
    for pred in predictions:
        pred['area'] = get_box_area(pred['corners'])
        length, width = get_obb_dimensions(pred['corners'])
        pred['length'] = length
        pred['width'] = width
    
    # Area-based filters
    if size_filter in ["largest", "biggest"]:
        predictions.sort(key=lambda x: x['area'], reverse=True)
        return predictions[:1]
    elif size_filter in ["smallest", "tiniest"]:
        predictions.sort(key=lambda x: x['area'])
        return predictions[:1]
    
    # Length-based filters (longer dimension)
    elif size_filter in ["longest", "lengthiest"]:
        predictions.sort(key=lambda x: x['length'], reverse=True)
        return predictions[:1]
    elif size_filter == "shortest":
        predictions.sort(key=lambda x: x['length'])
        return predictions[:1]
    
    # Width-based filters (shorter dimension)
    elif size_filter in ["widest", "broadest"]:
        predictions.sort(key=lambda x: x['width'], reverse=True)
        return predictions[:1]
    elif size_filter in ["narrowest", "thinnest"]:
        predictions.sort(key=lambda x: x['width'])
        return predictions[:1]
    
    return predictions


def filter_by_ordinal(predictions, ordinal, img_w, img_h):
    """Filter predictions by ordinal position."""
    if not predictions:
        return predictions
    
    ordinal = ordinal.lower()
    
    if ordinal in ["leftmost", "left-most", "left edge", "left-edge", "far left", "far-left", "left side", "left-side"]:
        predictions.sort(key=lambda x: get_box_center(x['corners'])[0])
        return predictions[:1]
    elif ordinal in ["rightmost", "right-most", "right edge", "right-edge", "far right", "far-right", "right side", "right-side"]:
        predictions.sort(key=lambda x: get_box_center(x['corners'])[0], reverse=True)
        return predictions[:1]
    elif ordinal in ["topmost", "top most", "top-most", "uppermost", "upper most", "top edge", "top-edge", "upper edge", "upper-edge", "upper-most"]:
        predictions.sort(key=lambda x: get_box_center(x['corners'])[1])
        return predictions[:1]
    elif ordinal in ["bottommost", "bottom most", "bottom-most", "lowermost", "lower most", "bottom edge", "bottom-edge", "lower edge", "lower-edge", "lower-most"]:
        predictions.sort(key=lambda x: get_box_center(x['corners'])[1], reverse=True)
        return predictions[:1]
    elif ordinal in ["first", "1st"]:
        return predictions[:1]
    elif ordinal in ["second", "2nd"] and len(predictions) > 1:
        return predictions[1:2]
    elif ordinal in ["third", "3rd"] and len(predictions) > 2:
        return predictions[2:3]
    elif ordinal in ["nearest", "closest"]:
        # Sort by distance from center of image
        center = (img_w / 2, img_h / 2)
        predictions.sort(key=lambda x: np.linalg.norm(get_box_center(x['corners']) - np.array(center)))
        return predictions[:1]
    elif ordinal in ["farthest", "furthest"]:
        center = (img_w / 2, img_h / 2)
        predictions.sort(key=lambda x: np.linalg.norm(get_box_center(x['corners']) - np.array(center)), reverse=True)
        return predictions[:1]
    
    return predictions


def filter_by_reference(predictions, reference_class, all_predictions):
    """Filter to find predictions closest to a reference object."""
    if not predictions:
        return predictions
    
    print("all pred")
    print(all_predictions)

    for p in all_predictions:
        print(p)
        print(p['class_name'])
    
    
    ref_objects = [p for p in all_predictions if p['class_name'] == reference_class]
    if not ref_objects:
        return predictions
    
    for pred in predictions:
        pred_center = get_box_center(pred['corners'])
        min_dist = float('inf')
        for ref in ref_objects:
            ref_center = get_box_center(ref['corners'])
            dist = np.sqrt((pred_center[0] - ref_center[0])**2 + 
                          (pred_center[1] - ref_center[1])**2)
            min_dist = min(min_dist, dist)
        pred['ref_distance'] = min_dist
    
    predictions.sort(key=lambda x: x['ref_distance'])
    return predictions[:1]


def filter_by_class(yolo_results, target_class):

    if(len(yolo_results)==0):
        return [],[]

    all_predictions=[]
    target_predictions=[]

    if yolo_results[0].obb is not None and len(yolo_results[0].obb) > 0:
        obb_data = yolo_results[0].obb
        for i in range(len(obb_data)):
            cls_id = int(obb_data.cls[i].cpu().numpy())
            corners = obb_data.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            confidence = float(obb_data.conf[i].cpu().numpy())
            
            pred = {
                'class_id': cls_id,
                'class_name': DIOR_CLASSES[cls_id],
                'confidence': confidence,
                'corners': corners,
            }
            all_predictions.append(pred)
            
            if DIOR_CLASSES[cls_id] == target_class:
                target_predictions.append(pred)
    
    return target_predictions, all_predictions