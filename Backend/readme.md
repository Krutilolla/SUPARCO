## Folder Structure
```bash
Backend/
│
├── public/                # Public static assets (e.g., uploaded / sample images)
│   └── image.jpg
│
├── Schema/                # Pydantic models for request/response validation
│   ├── input_json.py
│   ├── Output_json.py
│   └── Sessions.py
│
├── system_prompts/        # System prompt templates for different AI tasks
│   └── prompts.py
│
├── database.py            # DB connection & session handling
├── grounding.py           # GroundingDINO / grounding-related processing
├── helpers.py             # Helper utilities shared across modules
├── inference.py           # AI model inference logic
├── Main.py                # FastAPI entry point (server starts here)
│
├── Dockerfile             # Docker setup
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup (if needed)
├── .env                   # Environment variables
├── .dockerignore
└── .gitignore
```
## Backend Setup & Installation
## Through Docker
Run these command oderwise
```bash
docker build -t fast-api .
docker run -d -p 8000:8000 --name my-backend-container fastapi-app
```

## Run through virtual environment
### Create Virtual Environment
Make sure conda is installed in your system and 
create a virtual  environment name suparco
```bash
conda create --name suparco
```
```bash
conda activate suparco
```


### Installing Dependencies
```bash
pip install -r requirements.txt
```
```bash
uvicorn Main:app --host 0.0.0.0 --port 8000 --reload
```


## Environment Variables
Create a .env file  and configure it as per the mapping below
```bash
MONGO_URL=       Your mongodb Connection Uri
MODEL_SAVE_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/qwen
PROCESSOR_SAVE_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/processor
DINO_MODEL_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/dino_model
DINO_PROCESSOR_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/dino_processor
SAM_MODEL_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/sam_model
SAM_PROCESSOR_PATH=/home/Ubuntu/SUPARCO/Backend/model_weights/sam_processor
```
