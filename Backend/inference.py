import sys
import json
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import os
from qwen_vl_utils import process_vision_info


def load_qwen2vl(model_save_path, processor_save_path):
    print(model_save_path)
    print(processor_save_path)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    if os.path.exists(model_save_path) and os.listdir(model_save_path) and os.path.exists(processor_save_path) and os.listdir(processor_save_path):
        # Load from local path
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_save_path)
        processor = AutoProcessor.from_pretrained(processor_save_path)
        model.eval()
        print(model.device)
        return model, processor

    else:
        model_id =  "Qwen/Qwen2-VL-7B-Instruct" 
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf16",
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config,
            offload_folder="offload"
        )

        model.save_pretrained(model_save_path, from_pt=True) 
        processor.save_pretrained(processor_save_path)
        print(model.device)


        model.eval()
        return model, processor



def run_qwen(model, processor, system_prompt, image_local_path, questions,max_new_tokens=10):
    image = Image.open(image_local_path).convert("RGB")

    content = [{"type": "text", "text": question} for question in questions]
    content.append({"type": "image"})

    prompts = []

    # Build a separate conversation for each question
    for q in questions:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": q},
                    {"type": "image"}
                ]
            }
        ]

        # Convert into prompt text
        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # Process as batch (same image repeated; processor handles this)
    inputs = processor(
        text=prompts,
        images=[image] * len(prompts),   # repeat image for each question
        return_tensors="pt",
        padding=True
    ).to(model.device)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )

    generated_only = output_ids[:, inputs["input_ids"].shape[1]:]

    pred_answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True
    )
    print(pred_answer)

    del inputs, generated_only, output_ids
    torch.cuda.empty_cache()

    return pred_answer

def run_qwen_ft(model, processor, image_path, prompt_text, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del generated_ids, inputs
    torch.cuda.empty_cache()
    
    # Split to get only the assistant's new reply
    if "assistant\n" in output_text:
        return output_text.split("assistant\n")[-1]
    return output_text.strip()




