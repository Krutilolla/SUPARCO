from system_prompts.prompts import captioning_system_prompt, vqa_system_prompt
from inference import run_qwen
from vqa_pipeline import route_vqa_query
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grounding import run_grounding


def handle_captioning(model, processor, image_local_path, caption_prompt="Describe image in detail."):
    responses = run_qwen(model=model, processor=processor, system_prompt=captioning_system_prompt,image_local_path=image_local_path, questions=[caption_prompt], max_new_tokens=300)
    return responses

def handle_vqa(model, processor, image_local_path, questions, llm_model, llm_tokenizer, yolo_model):
    questions_list = [q for q in questions.values() if q]
    print(questions_list)
    responses = []
    for question in questions_list:
        response = route_vqa_query(question, image_local_path, llm_model, llm_tokenizer, model, processor, yolo_model)
        responses.append(response)
    # responses = run_qwen(model=model,processor=processor,system_prompt=vqa_system_prompt,image_local_path=image_local_path, questions=questions_list)
    return responses

def handle_vqa_frontend(model, processor, image_local_path, question):
    responses = run_qwen(model=model,processor=processor,system_prompt=vqa_system_prompt,image_local_path=image_local_path, questions=[question])
    return responses

def handle_grounding(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, yolo_model, image_local_path, text_prompt, llm_model, llm_tokenizer):
    responses, obb_image_path = run_grounding(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, yolo_model, image_local_path, text_prompt, llm_model, llm_tokenizer)
    print(responses)

    return responses, obb_image_path

    

    
    