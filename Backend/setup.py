from transformers import (
    GroundingDinoProcessor, GroundingDinoForObjectDetection,
    SamProcessor, SamModel,
    AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
)

from peft import PeftModel

import torch
import os
from ultralytics import YOLO

def load_grounding_models(dino_model_path, dino_processor_path, sam_model_path, sam_processor_path):

    print(dino_model_path)
    print(dino_processor_path)
    print(sam_model_path)
    print(sam_processor_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.path.exists(dino_model_path) and os.listdir(dino_model_path) and os.path.exists(dino_processor_path) and os.listdir(dino_processor_path) and os.path.exists(sam_model_path) and os.listdir(sam_model_path) and os.path.exists(sam_processor_path) and os.listdir(sam_processor_path):
        grounding_dino_processor = GroundingDinoProcessor.from_pretrained(dino_processor_path)
        grounding_dino_model = GroundingDinoForObjectDetection.from_pretrained(dino_model_path).to(device)

        sam_processor = SamProcessor.from_pretrained(sam_processor_path)
        sam_model = SamModel.from_pretrained(sam_model_path).to(device)
        return grounding_dino_model, grounding_dino_processor, sam_model, sam_processor

    else:
        grounding_dino_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        grounding_dino_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

        # print(grounding_dino_model, grounding_dino_processor)
        grounding_dino_model.save_pretrained(dino_model_path)
        grounding_dino_processor.save_pretrained(dino_processor_path)
        sam_model.save_pretrained(sam_model_path)
        sam_processor.save_pretrained(sam_processor_path)
        return grounding_dino_model, grounding_dino_processor, sam_model, sam_processor


def load_yolo(yolo_model_path):
    yolo_model = YOLO(yolo_model_path)
    return yolo_model

def load_llm(llm_model_path, llm_tokenizer_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.path.exists(llm_model_path) and os.listdir(llm_model_path) and os.path.exists(llm_tokenizer_path) and os.listdir(llm_tokenizer_path):
        llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_path)
        return llm_model, llm_tokenizer

    else:

        llm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16,
        )
        llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        llm_model.save_pretrained(llm_model_path)
        llm_tokenizer.save_pretrained(llm_tokenizer_path)
        return llm_model, llm_tokenizer


def load_ft_model(base_model_id, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        trust_remote_code = True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code = True)
    model.eval()
    return model, processor