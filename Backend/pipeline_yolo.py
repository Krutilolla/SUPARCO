import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU not available! Please enable GPU in Kaggle: Settings → Accelerator → GPU")

# print(f"CUDA version: {torch.version.cuda}")
# print(f"GPU: {torch.cuda.get_device_name(0)}")

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid

import torch
from PIL import Image
from pathlib import Path
import json


from system_prompts.yolo_filter_rules import *
from system_prompts.prompts import llm_get_filters_grounding_prompt
llm_model=None
llm_tokenizer=None

import re
from difflib import SequenceMatcher
from grounding_query_parser import *
from grounding_filters import *


# ============================================================
# MAIN GROUNDING FUNCTION (SINGLE CLASS)
# ============================================================
# def advanced_grounding_predict(image_path, prompt, model, class_list, conf=0.25, color_threshold=0.10, query):
#     """
#     ROBUST visual grounding with:
#     - Synonym mapping (plane → airplane, car → vehicle)
#     - Fuzzy matching for typos
#     - Spatial/size/ordinal/color/reference filters
    
#     Handles prompts like:
#         - "Find the plane" → airplane
#         - "The biggest car in the top left" → largest vehicle in top left
#         - "The white aeroplane in the corner" → white airplane in corner
#         - "The boat near the port" → ship near harbor
    
#     Args:
#         image_path: Path to image
#         prompt: Natural language query
#         model: YOLO OBB model
#         class_list: List of class names
#         conf: Confidence threshold
#         color_threshold: Color matching threshold
    
#     Returns:
#         List of prediction dicts
#     """
#     # Parse prompt
#     # query = parse_grounding_prompt(prompt)
    
#     target_class = query['class']
    
#     if not target_class:
#         return []
    
#     # print(f"   🎯 Target class: {target_class}")
    
#     # Show filters
#     filters = []
#     if query.get('spatial'): filters.append(f"spatial={query['spatial']}")
#     if query.get('size'): filters.append(f"size={query['size']}")
#     if query.get('color'): filters.append(f"color={query['color']}")
#     if query.get('ordinal'): filters.append(f"ordinal={query['ordinal']}")
#     if query.get('reference'): filters.append(f"reference={query['reference']}")
#     if filters:
#         print(f"Filters: {', '.join(filters)}")
    
#     # Load image
#     img_bgr = cv2.imread(str(image_path))
#     if img_bgr is None:
#         return []
#     img_h, img_w = img_bgr.shape[:2]
    
#     # Run YOLO
#     results = model.predict(image_path, imgsz=800, conf=conf, verbose=False, save=False)
#     # print("YOLO RAW Results", results)

#     target_predictions, all_predictions = filter_by_class(results, class_list, target_class)
#     filtered = filter_boxes(target_predictions, query, all_predictions, img_bgr, img_w, img_h, color_threshold)
    
#     return filtered

def run_gdino_sam_pipeline(image_path, prompt, gdino_processor, gdino_model, 
                           sam_processor, sam_model, device, threshold=0.25):
    """
    Run GroundingDINO + SAM + minAreaRect pipeline.
    Returns list of OBB predictions with 'corners' key (4x2 numpy arrays)
    """
    image = Image.open(image_path).convert("RGB")
    
    inputs = gdino_processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = gdino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, target_sizes=target_sizes
    )
    
    all_aabb_boxes = results[0]['boxes']
    all_scores = results[0]['scores']
    
    if len(all_aabb_boxes) == 0:
        return []
    
    all_obb_predictions = []
    
    for idx, aabb_box in enumerate(all_aabb_boxes):
        try:
            input_boxes = [[aabb_box.tolist()]]
            sam_inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)
            
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
            
            reshaped_hw = sam_inputs["pixel_values"].shape[2:]
            reshaped_input_sizes = torch.tensor([reshaped_hw]).to(device)
            
            mask = sam_processor.post_process_masks(
                sam_outputs.pred_masks,
                sam_inputs["original_sizes"].to(device),
                reshaped_input_sizes,
                binarize=True
            )[0]
            
            binary_mask_cv = (mask[0][0].cpu().numpy() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                oriented_box = cv2.minAreaRect(largest_contour)
                box_points = cv2.boxPoints(oriented_box).astype(np.float32)
                
                all_obb_predictions.append({
                    'corners': box_points,
                    'confidence': float(all_scores[idx].cpu().numpy()),
                    'class_name': 'gdino_detection',
                    'source': 'gdino+sam'
                })
        except Exception as e:
            # print(f"   ⚠️ SAM failed for box {idx}: {e}")
            continue
    
    return all_obb_predictions


# ============================================================
# HELPER: YOLO OBB Detection with Filters
# (Same logic as advanced_grounding_predict from Section 6)
# ============================================================
def run_yolo_with_filters(image_path, query, model, class_list, conf=0.25, color_threshold=0.10):
    """
    Run YOLO OBB detection with all filters applied.
    Uses the SAME logic as advanced_grounding_predict() from Section 6.
    """
    target_class = query.get('class')
    
    # Same check as original - only check if target_class exists
    if not target_class:
        return []
    
    
    # Load image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return []
    img_h, img_w = img_bgr.shape[:2]
    
    # Run YOLO
    results = model.predict(image_path, imgsz=800, conf=conf, verbose=False, save=False)
    return results


# ============================================================
# MAIN UNIFIED PIPELINE
# ============================================================
def unified_grounding_pipeline(
    image_path,
    prompt,
    yolo_model,
    class_list,
    llm_model,
    llm_tokenizer,
    query,
    gdino_processor=None,
    gdino_model=None,
    sam_processor=None,
    sam_model=None,
    device="cuda",
    conf=0.25,
    color_threshold=0.10,
    gdino_threshold=0.25,
    run_parallel_gdino=False,
    
):
    """
    🚀 UNIFIED VISUAL GROUNDING PIPELINE
    
    Flow:
    1. Rule-based keyword extraction (parse_grounding_prompt)
    2. If no class found → LLM fallback (Qwen2.5)
    3. If still no class → GDINO+SAM fallback (placeholder)
    4. YOLO OBB detection with filters
    5. Parallel GDINO+SAM processing
    6. Combined visualization
    
    Args:
        image_path: Path to input image
        prompt: Natural language query
        yolo_model: YOLO OBB model
        class_list: DIOR-R class list
        gdino_processor, gdino_model: GroundingDINO components
        sam_processor, sam_model: SAM components
        device: torch device
        conf: YOLO confidence threshold
        color_threshold: Color filter threshold
        gdino_threshold: GDINO detection threshold
        run_parallel_gdino: Whether to run GDINO in parallel
    
    Returns:
        dict with 'yolo_predictions', 'gdino_predictions', 'pipeline_used', 'query'
    """
    
    result = {
        'yolo_predictions': [],
        'gdino_predictions': [],
        'pipeline_used': None,
        'query': None
    }
    
    result['query'] = query
    
    # ─────────────────────────────────────────────────────────
    # STEP 4: YOLO Detection (if class found)
    # ─────────────────────────────────────────────────────────
    if query.get('class'):
        
        yolo_preds = run_yolo_with_filters(
            image_path, query, yolo_model, class_list,
            conf=conf, color_threshold=color_threshold
        )
        result['yolo_predictions'] = yolo_preds
        # print(f"   ✅ YOLO found {len(yolo_preds)} filtered object(s)")
    
    # ─────────────────────────────────────────────────────────
    # STEP 5: Parallel GDINO+SAM (if enabled and models available)
    # ─────────────────────────────────────────────────────────
    if run_parallel_gdino and gdino_processor and gdino_model and sam_processor and sam_model:
        # # print(f"\n{'─'*50}")
        # # print("🔍 STEP 5: Parallel GroundingDINO + SAM")
        # # print(f"{'─'*50}")
        
        gdino_preds = run_gdino_sam_pipeline(
            image_path, prompt,
            gdino_processor, gdino_model,
            sam_processor, sam_model,
            device, threshold=gdino_threshold
        )
        result['gdino_predictions'] = gdino_preds
        # # print(f"   ✅ GDINO+SAM found {len(gdino_preds)} object(s)")
    elif run_parallel_gdino:
        print(f"GDINO/SAM models not loaded, skipping parallel detection")
    
    return result


# ============================================================
# 🎯 RUN UNIFIED PIPELINE - EXAMPLE
# ============================================================

def generate_obb_yolo(image_path, yolo_model, prompt, llm_model, llm_tokenizer, query):
    # global llm_model, llm_tokenizer
    
    # llm_model=llm_model_arg
    # llm_tokenizer=llm_tokenizer_arg
    

    result = unified_grounding_pipeline(
        image_path=str(image_path),
        prompt=prompt,
        yolo_model=yolo_model,
        class_list=DIOR_CLASSES,
        gdino_processor=grounding_dino_processor if 'grounding_dino_processor' in dir() else None,
        gdino_model=grounding_dino_model if 'grounding_dino_model' in dir() else None,
        sam_processor=sam_processor if 'sam_processor' in dir() else None,
        sam_model=sam_model if 'sam_model' in dir() else None,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        query=query,
        device=device if 'device' in dir() else "cuda",
        conf=0.3,
        run_parallel_gdino=False  # Set to False if GDINO/SAM not loaded
    )

    return result
    

