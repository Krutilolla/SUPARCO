import cv2
import glob
import json
import numpy as np
import os
import time
import torch
from numeric_helper import get_target_class_from_qwen, count_yolo_detections 
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from system_prompts.prompts import vqa_system_prompt
from inference import run_qwen, run_qwen_ft

device = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
    'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station', 
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]


def handle_numeric_vqa(question, image_path_vqa, model, tokenizer,qwen_vqa,processor, yolo_model):
    
    yolo_out = yolo_model.predict(image_path_vqa, conf=0.25, verbose=False)
    class_of_object = get_target_class_from_qwen(question, tokenizer, model)
    object_class = str(class_of_object)
    print(object_class)
    if "SUPARCO" in object_class.upper() : 
        return handle_full_vqa(question,image_path_vqa,qwen_vqa,processor)
    count = count_yolo_detections(yolo_out[0], class_of_object)
    print("YOLO Count Result", count)
    return count

def handle_full_vqa(question, image_path_vqa, model, processor):
    prompt = vqa_system_prompt

    # pred = run_qwen(model, processor, vqa_system_prompt, image_path_vqa, [question])

    # return pred[0]
    pred = run_qwen_ft(model, processor, image_path_vqa, question, vqa_system_prompt)

    return pred




def route_vqa_query(question, image_path_vqa, model, tokenizer, qwen_vqa, processor, yolo_model):
    """
    Uses Qwen 1.5B to classify the question and routes it to the appropriate handler.
    """
    system_prompt = (
        "You are a query classifier for satellite image analysis. "
        "Classify the user question into exactly one of these three categories:\n"
        "1. 'numeric' - for questions asking for counts, quantities, or numbers (e.g., 'How many...').\n"
        "2. 'binary' - for questions with Yes/No answers or existence checks (e.g., 'Is there...', 'Are there...').\n"
        "3. 'semantic' - for open-ended questions about object categories, descriptions, or attributes (e.g., 'What is...', 'Describe...').\n\n"
        "Return ONLY the category name: 'numeric', 'binary', or 'semantic'."
    )

    user_prompt = f"Question: \"{question}\"\nCategory:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Prepare inputs
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate classification token
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.01, # Low temperature for deterministic behavior
            do_sample=False
        )
    
    # Extract only the new tokens (the response)
    input_length = model_inputs.input_ids.shape[1]
    response_tokens = generated_ids[:, input_length:]
    classification = tokenizer.decode(response_tokens[0], skip_special_tokens=True).strip().lower()
    
    # Routing Logic
    if "numeric" in classification:
        return handle_numeric_vqa(question, image_path_vqa, model, tokenizer,qwen_vqa,processor, yolo_model)
        print(f"Sent to Numeric Handler")
    else:
        return handle_full_vqa(question, image_path_vqa, qwen_vqa, processor)
        print(f"Sent to Binary Semantic Handler")
    



