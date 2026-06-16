from setup import *
import os
from dotenv import load_dotenv
from helpers import handle_grounding

load_dotenv()

dino_model_path = os.getenv("DINO_MODEL_PATH")
dino_processor_path = os.getenv("DINO_PROCESSOR_PATH")
sam_model_path = os.getenv("SAM_MODEL_PATH")
sam_processor_path = os.getenv("SAM_PROCESSOR_PATH")
yolo_model_path = os.getenv("YOLO_MODEL_PATH")
llm_model_path=os.getenv("LLM_MODEL_PATH")
llm_tokenizer_path=os.getenv("LLM_TOKENIZER_PATH")

grounding_dino_model, grounding_dino_processor, sam_model, sam_processor = load_grounding_models(dino_model_path, dino_processor_path, sam_model_path, sam_processor_path)
yolo_model = load_yolo(yolo_model_path)
llm_model, llm_tokenizer=load_llm(llm_model_path, llm_tokenizer_path)

image_local_path="/home/Ubuntu/SUPARCO/Backend/public/windmill.jpg"
text_prompt="Locate windmills in the image"

results = handle_grounding(grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, yolo_model, image_local_path, text_prompt, llm_model, llm_tokenizer)