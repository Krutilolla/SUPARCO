from ultralytics import YOLO
import os
import torch

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")

YOLO_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
    'chimney', 'dam', 'expressway-service-area', 'expressway-toll-station', 
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
]

def get_target_class_from_qwen(question, tokenizer, model):
    system_prompt = (
        "You are an intelligent assistant for satellite imagery analysis. "
        "Your task is to identify the object class the user wants to count from t"
        f"You MUST map the object to exactly one of these supported classes: [{YOLO_CLASSES}] "  
        "Return ONLY the single class name from the list. Do not write a sentence. "
        "If No Matching Classes are found strictly print SUPARCO nothing else , DO NOT Return the original MisMatched Class ONLY SUPARCO "
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: '{question}'\nTarget Class:"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
    return response.split("assistant")[-1].strip().lower()

def count_yolo_detections(result, target_class):
    # OBB vs Standard check
    if result.obb is not None:
        detections = result.obb
    elif result.boxes is not None:
        detections = result.boxes
    else:
        return 0
        
    clean_target = target_class.replace(" ", "").replace("-", "")
    count = 0
        
    for box in detections:
        cls_id = int(box.cls[0].item())
        name = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)
        clean_name = name.lower().replace(" ", "").replace("-", "")
        
        if clean_target in clean_name or clean_name in clean_target:
            count += 1
                
    return str(count)

    
