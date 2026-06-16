from grounding_dino_pipeline import generate_obb_pipeline
from pipeline_yolo import generate_obb_yolo
from bboxes_helper import evaluate_grounding_score, best_boxes
from grounding_filters import *
from grounding_query_parser import *
from PIL import Image
import cv2
import numpy as np

BBOX_SAVE_PATH="/home/Ubuntu/SUPARCO/Backend/public/image_with_obb_overlapping.jpeg"

def run_grounding(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, yolo_model, image_local_path, text_prompt, llm_model, llm_tokenizer, color_threshold=0.10):
    query = parse_grounding_prompt(text_prompt, llm_model, llm_tokenizer)
    print(query)
    target_class = query.get('class')
    
    gd_boxes, obb_image_path = generate_obb_pipeline(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, image_local_path, text_prompt)
    print(gd_boxes)

    yolo_results = generate_obb_yolo(image_local_path, yolo_model,text_prompt, llm_model, llm_tokenizer, query)
    
    print(yolo_results)


    target_predictions, all_predictions = filter_by_class(yolo_results['yolo_predictions'], target_class)
    print(target_predictions)

    img_bgr = cv2.imread(str(image_local_path))
    if img_bgr is None:
        return []
    img_h, img_w = img_bgr.shape[:2]

    filtered_boxes_yolo = filter_boxes(target_predictions, query, all_predictions, img_bgr, img_w, img_h, color_threshold)
    print(filtered_boxes_yolo)

    all_boxes_yolo = [pred['corners'].tolist() for pred in all_predictions]
    image_final = Image.open(image_local_path).convert("RGB")
    image_final = cv2.cvtColor(np.array(image_final), cv2.COLOR_RGB2BGR)

    for box in all_boxes_yolo:
        obb = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_final, [obb], isClosed=True, color=(0, 0, 255), thickness=3) 
    
    cv2.imwrite("/home/Ubuntu/SUPARCO/Backend/public/image_with_obb_yolo.jpeg", image_final)

    gd_boxes_results = [{'corners': np.array(gd_box)} for gd_box in gd_boxes]
    print(gd_boxes_results)

    filtered_boxes_gd = filter_boxes(gd_boxes_results, query, gd_boxes_results, img_bgr, img_w, img_h, color_threshold)
    print(filtered_boxes_gd)

    yolo_bbox = [pred['corners'].tolist() for pred in filtered_boxes_yolo]
    gd_bbox = [pred['corners'].tolist() for pred in filtered_boxes_gd]

    image_final = Image.open(image_local_path).convert("RGB")
    image_final = cv2.cvtColor(np.array(image_final), cv2.COLOR_RGB2BGR)

    for box in gd_bbox:
        obb = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_final, [obb], isClosed=True, color=(0, 0, 255), thickness=3) 

    for box in yolo_bbox:
        obb = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_final, [obb], isClosed=True, color=(0, 255, 0), thickness=3) 
    
    cv2.imwrite("/home/Ubuntu/SUPARCO/Backend/public/image_with_obb_yolo_dino.jpeg", image_final)
    
    image_overlapping = Image.open(image_local_path).convert("RGB")
    image_overlapping = cv2.cvtColor(np.array(image_overlapping), cv2.COLOR_RGB2BGR)
    overlapping_boxes = best_boxes(yolo_bbox, gd_bbox)

    for box in overlapping_boxes:
        obb = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_overlapping, [obb], isClosed=True, color=(0, 255, 0), thickness=3) 
    
    cv2.imwrite(BBOX_SAVE_PATH, image_overlapping)

    
    return overlapping_boxes, BBOX_SAVE_PATH

def filter_boxes(boxes_predictions, query, all_predictions, img_bgr, img_w, img_h, color_threshold):
    # Apply filters
    # Show filters being applied
    filters = []
    if query.get('spatial'): filters.append(f"spatial={query['spatial']}")
    if query.get('size'): filters.append(f"size={query['size']}")
    if query.get('color'): filters.append(f"color={query['color']}")
    if query.get('ordinal'): filters.append(f"ordinal={query['ordinal']}")
    if query.get('reference'): filters.append(f"reference={query['reference']}")
    if filters:
        print(f"Filters: {', '.join(filters)}")
    
    filtered = boxes_predictions.copy()
    
    if query.get('color'):
        filtered = filter_by_color(filtered, query['color'], img_bgr, threshold=color_threshold)
        # print(f"   After color filter ({query['color']}): {len(filtered)} objects")
    
    if query.get('spatial'):
        filtered = filter_by_spatial_region(filtered, query['spatial'], img_w, img_h)
        # print(f"   After spatial filter ({query['spatial']}): {len(filtered)} objects")
    
    if query.get('size'):
        filtered = filter_by_size(filtered, query['size'])
        # print(f"   After size filter ({query['size']}): {len(filtered)} objects")
    
    if query.get('reference') and len(filtered)>0 and 'class_name' in list(filtered[0].keys()):
        filtered = filter_by_reference(filtered, query['reference'], all_predictions)
        # print(f"   After reference filter (near {query['reference']}): {len(filtered)} objects")
    
    if query.get('ordinal'):
        filtered = filter_by_ordinal(filtered, query['ordinal'], img_w, img_h)
    
    return filtered

