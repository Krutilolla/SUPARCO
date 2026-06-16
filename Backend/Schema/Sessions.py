from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import datetime


class Chats(BaseModel):
    message: str
    sender: str
    task: str
    url:Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionCreate(BaseModel):
    name: str
    imageurl: Optional[HttpUrl] = None
    chats: List[Chats] = Field(default_factory=list)
    createdat: datetime = Field(default_factory=datetime.utcnow)
    updatedat: datetime = Field(default_factory=datetime.utcnow)


class SessionInDB(SessionCreate):
    id: str = Field(alias="_id")


class ImageUpdate(BaseModel):
    imageurl: HttpUrl


class ChatRequest(BaseModel):
    message: str
    task:str
