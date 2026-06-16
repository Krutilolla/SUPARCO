from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Any, Dict
from bson import ObjectId
from Schema.input_json import FullRequest
import requests
from bson import ObjectId
from inference import load_qwen2vl
from helpers import handle_captioning, handle_grounding, handle_vqa, handle_vqa_frontend
from setup import load_grounding_models, load_yolo, load_llm, load_ft_model

from database import collection
from Schema.Sessions import SessionCreate, SessionInDB, ImageUpdate, ChatRequest, Chats

import os
from datetime import datetime
import cloudinary
import cloudinary.uploader
import cloudinary.api
from Schema.Output_json import FullResponse,build_final_response
import cv2

cloudinary.config( 
  cloud_name = "dzczys4gk", 
  api_key = "156298573688467", 
  api_secret = "-D9g5xaALh8hoV8I2wwwshoJ9d0"
)
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


@router.get("/")
async def home() -> Dict[str, str]:
    return {"message": "Hello"}


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "OK"}


@router.post("/create_session")
async def create_session(session: SessionCreate) -> Any:
    session_dict = session.model_dump(mode="json")  # 👈 important
    result = collection.insert_one(session_dict)
    created = collection.find_one({"_id": result.inserted_id})
    if created and "_id" in created:
        created["_id"] = str(created["_id"])
    return SessionInDB(**created)



@router.patch("/update_session/{session_id}")
async def update_session_image(session_id: str, payload: ImageUpdate) -> Dict[str, str]:
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # 🔥 Convert payload to JSON-safe types
    update_data = payload.model_dump(mode="json")

    update_result = collection.update_one(
        {"_id": oid},
        {
            "$set": {
                **update_data,   # includes "imageurl" as a string now
                "updatedat": datetime.utcnow()
            }
        },
    )

    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session updated successfully"}





@router.get("/get_session")
async def get_sessions():
    sessions = []
    for doc in collection.find():
        doc["_id"] = str(doc["_id"])
        sessions.append(doc)

    return sessions

# Load the model once when the server starts running, add this part to the relevant function
# model, processor = load_qwen2vl()
model=None
processor=None
grounding_dino_model, grounding_dino_processor, sam_model, sam_processor = None, None, None, None
yolo_model = None
llm_model, llm_tokenizer = None, None
qwen_ft_model, qwen_ft_processor = None, None
@app.on_event("startup")
async def startup_event():
    global model, processor, grounding_dino_model, grounding_dino_processor, sam_model, sam_processor, yolo_model, llm_model, llm_tokenizer, qwen_ft_model, qwen_ft_processor
    model_save_path = os.getenv("MODEL_SAVE_PATH")
    processor_save_path = os.getenv("PROCESSOR_SAVE_PATH")
    dino_model_path = os.getenv("DINO_MODEL_PATH")
    dino_processor_path = os.getenv("DINO_PROCESSOR_PATH")
    sam_model_path = os.getenv("SAM_MODEL_PATH")
    sam_processor_path = os.getenv("SAM_PROCESSOR_PATH")
    yolo_model_path = os.getenv("YOLO_MODEL_PATH")
    llm_model_path = os.getenv("LLM_MODEL_PATH")
    llm_tokenizer_path = os.getenv("LLM_TOKENIZER_PATH")
    adapter_path=os.getenv("ADAPTER_PATH")
    model,processor = load_qwen2vl(model_save_path, processor_save_path)
    grounding_dino_model, grounding_dino_processor, sam_model, sam_processor = load_grounding_models(dino_model_path, dino_processor_path, sam_model_path, sam_processor_path)
    yolo_model = load_yolo(yolo_model_path)
    llm_model, llm_tokenizer = load_llm(llm_model_path, llm_tokenizer_path)
    qwen_ft_model, qwen_ft_processor = load_ft_model(base_model_id="Qwen/Qwen2.5-VL-3B-Instruct", adapter_path=adapter_path)
    # print(qwen_ft_model, qwen_ft_processor)
    

@router.post("/eval")
async def process_json(payload: FullRequest):
    # ----------------------------------------
    # 1. DOWNLOAD IMAGE SAFELY
    # ----------------------------------------
    url = payload.input_image.image_url
    if not url:
        raise HTTPException(status_code=400, detail="input_image.image_url is required")

    try:
        imgresponse = requests.get(url, timeout=15)
        imgresponse.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

    image_path = "./public/image.jpg"
    with open(image_path, "wb") as f:
        f.write(imgresponse.content)

    # ----------------------------------------
    # Initialize placeholders for response
    # ----------------------------------------
    caption_text = None
    grounding_objects = []   # list of {"object_id": str, "obbox": [x, y, ...]}
    binary_instruction = binary_response = None
    numeric_instruction = numeric_response = None
    semantic_instruction = semantic_response = None

    width = payload.input_image.metadata.width
    height = payload.input_image.metadata.height
    resolution = payload.input_image.metadata.spatial_resolution_m

    # ----------------------------------------
    # 2. CAPTION QUERY
    # ----------------------------------------
    try:
        caption_query = payload.queries.caption_query
        if caption_query and caption_query.instruction:
            caption_result = handle_captioning(
                model=model,
                processor=processor,
                image_local_path=image_path,
                caption_prompt=caption_query.instruction
            )

            # handle_captioning returns list → extract first
            if isinstance(caption_result, (list, tuple)) and caption_result:
                caption_text = caption_result[0]
            else:
                caption_text = caption_result

    except Exception as e:
        print("Caption error:", e)

    # ----------------------------------------
    # 3. ATTRIBUTE QUERY (VQA)
    # ----------------------------------------
    try:
        attribute_query = payload.queries.attribute_query
        if attribute_query:

            # Binary
            if attribute_query.binary and attribute_query.binary.instruction:
                binary_instruction = attribute_query.binary.instruction
                vqa_binary = handle_vqa(
                    # model=model,
                    # processor=processor,
                    model=qwen_ft_model,
                    processor=qwen_ft_processor,
                    image_local_path=image_path,
                    questions={"binary": binary_instruction},
                    llm_model = llm_model,
                    llm_tokenizer=llm_tokenizer,
                    yolo_model=yolo_model
                )
                binary_response=vqa_binary[0]
                # handle_vqa should return a dict or value; adapt as needed
                # binary_response = resp.get("binary") if isinstance(resp, dict) else resp

            if attribute_query.numeric and attribute_query.numeric.instruction:
                numeric_instruction = attribute_query.numeric.instruction
                vqa_numeric = handle_vqa(
                    # model=model,
                    # processor=processor,
                    model=qwen_ft_model,
                    processor=qwen_ft_processor,
                    image_local_path=image_path,
                    questions={"numeric": numeric_instruction},
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    yolo_model=yolo_model
                )
                numeric_response=vqa_numeric[0]
                # numeric_response = resp.get("numeric") if isinstance(resp, dict) else resp

            if attribute_query.semantic and attribute_query.semantic.instruction:
                semantic_instruction = attribute_query.semantic.instruction
                vqa_semantic = handle_vqa(
                    # model=model,
                    # processor=processor,
                    model=qwen_ft_model,
                    processor=qwen_ft_processor,
                    image_local_path=image_path,
                    questions={"semantic": semantic_instruction},
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    yolo_model=yolo_model
                )
                semantic_response=vqa_semantic[0]
                # semantic_response = resp.get("semantic") if isinstance(resp, dict) else resp
    except Exception as e:
        print("VQA error:", e)

    # ----------------------------------------
    # 4. GROUNDING (OBB DETECTION)
    # ----------------------------------------
    obb_image_cloud_url = None
    try:
        grounding_query = payload.queries.grounding_query
        if grounding_query and grounding_query.instruction:
            grounding_result, obb_image_path = handle_grounding(
                grounding_dino_model=grounding_dino_model,
                grounding_dino_processor=grounding_dino_processor,
                sam_model=sam_model,
                sam_processor=sam_processor,
                image_local_path=image_path,
                text_prompt=grounding_query.instruction,
                yolo_model=yolo_model,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer
            )

            print("\nGrounding Result\n")
            print(grounding_result)

                
            grounding_objects = [
            {
                "object_id": f"{i+1}",
                "obbox": [
                    coord / (width if j % 2 == 0 else height)
                    for point in box
                    for j, coord in enumerate(point)
                ]
            }
            for i, box in enumerate(grounding_result)
            ]
            print("Grounding Objects")
            print(grounding_objects)

            # Upload OBB visualization image
            if obb_image_path:
                try:
                    public_id = f"obb_{payload.input_image.image_id}_{int(datetime.utcnow().timestamp())}"
                    upload_data = cloudinary.uploader.upload(
                        obb_image_path,
                        folder="grounding_outputs",
                        public_id=public_id
                    )
                    obb_image_cloud_url = upload_data.get("secure_url")
                except Exception as e:
                    print("Cloudinary upload failed:", e)

    except Exception as e:
        print("Grounding error:", e)

    # ----------------------------------------
    # 5. FINAL RESPONSE
    # ----------------------------------------
    response = build_final_response(
        image_id=payload.input_image.image_id,
        image_url=str(payload.input_image.image_url),
        width=width,
        height=height,
        resolution=resolution,
        caption_query=caption_query.instruction,
        caption_text=caption_text,
        grounding_query=grounding_query.instruction,
        grounding_objects=grounding_objects,

        binary_instruction=binary_instruction,
        binary_response=binary_response,

        numeric_instruction=numeric_instruction,
        numeric_response=numeric_response,

        semantic_instruction=semantic_instruction,
        semantic_response=semantic_response,
    )


    # Optionally attach OBB image URL
    # if obb_image_cloud_url:
    #     response["queries"]["grounding_query"]["obb_image_url"] = obb_image_cloud_url

    print(response)

    # if hasattr(response, "model_dump"):
    # response = response.model_dump(mode="json")
    # print(response)

    return JSONResponse(response)


@router.post("/chat/{session_id}")
async def add_chat(session_id: str, chat: ChatRequest):
    print("task to be performed", chat.task)

    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # FIX: include task because Chats model requires it
    user_chat = Chats(
        message=chat.message,
        sender="user",
        task=chat.task
    )
    
    url=""
    doc =  collection.find_one({"_id": ObjectId(session_id)})
    if doc:
        url = doc.get("imageurl")
    else:
        url = None
    
    # url = .input_image.image_url
    imgresponse = requests.get(url)

    image_path = "./public/image.jpg"
    bot_response_text=""
    grounding_img_path=""
    with open(image_path, "wb") as f:
        f.write(imgresponse.content)
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    if(chat.task=="Generate Caption"):
        result = handle_captioning(
            model=model,
            processor=processor,
            image_local_path=image_path,
            caption_prompt = chat.message
        )
        print(result)
        bot_response_text = result[0]
        url=""

    
    elif(chat.task=="Visual Question Answering"):
        result = handle_vqa(
            # model=model,
            # processor=processor,
            model=qwen_ft_model,
            processor=qwen_ft_processor,
            image_local_path=image_path,
            questions={"question": chat.message},
            llm_model = llm_model,
            llm_tokenizer=llm_tokenizer,
            yolo_model=yolo_model
        )
        print(result)
        bot_response_text=result[0]
        url=""
        print("answer ", str(bot_response_text))

    elif(chat.task=="Generate grounding"):
        result, obb_image_path = handle_grounding(
                grounding_dino_model=grounding_dino_model,
                grounding_dino_processor=grounding_dino_processor,
                sam_model=sam_model,
                sam_processor=sam_processor,
                image_local_path=image_path,
                text_prompt=chat.message,
                yolo_model=yolo_model,
                llm_model=llm_model,
                llm_tokenizer=llm_tokenizer
            )
        grounding_objects = [
            {
                "object_id": f"{i+1}",
                "obbox": [
                    coord / (width if j % 2 == 0 else height)
                    for point in box
                    for j, coord in enumerate(point)
                ]
            }
            for i, box in enumerate(result)
        ]
        bot_response_text=f'{grounding_objects}'

        if obb_image_path is None:
            return {
            "status": "success",
            "user_message": chat.message,
            "response": "No Objects detected",
            }  

        else:
            grounding_img_path= cloudinary.uploader.upload(
            obb_image_path,
            folder="public",
            public_id="img"
            )
            print(grounding_img_path)
            print(grounding_img_path["secure_url"])
            url=grounding_img_path["secure_url"]

    #    print(result)
    #    print(obb_image_path)
       
       
    # print(bot_response_text[0])

    bot_chat = Chats(
        message=str(bot_response_text),
        sender="bot",
        task=chat.task,  # you can change if needed
        url=url
    )
    print(bot_chat.url)


    update_result = collection.update_one(
        {"_id": oid},
        {
            "$push": {
                "chats": {
                    "$each": [
                        user_chat.model_dump(),
                        bot_chat.model_dump(),
                    ]
                }
            },
            "$set": {"updatedat": datetime.utcnow()},
        },
    )
    print(update_result)

    if update_result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "status": "success",
        "user_message": chat.message,
        "response": bot_response_text,
        "image_url":url
    }

app.include_router(router)
