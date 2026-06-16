from pydantic import BaseModel
from typing import Optional



class Metadata(BaseModel):
    width: int
    height: int
    spatial_resolution_m: float

class CaptionQuery(BaseModel):
    instruction: str

class GroundingQuery(BaseModel):
    instruction: str

class BinaryQuery(BaseModel):
    instruction: str

class NumericQuery(BaseModel):
    instruction: str

class SemanticQuery(BaseModel):
    instruction: str

class AttributeQuery(BaseModel):
    binary: Optional[BinaryQuery]
    numeric: Optional[NumericQuery]
    semantic: Optional[SemanticQuery]

class Queries(BaseModel):
    caption_query: CaptionQuery
    grounding_query: GroundingQuery
    attribute_query: AttributeQuery

class InputImage(BaseModel):
    image_id: str
    image_url: str
    metadata: Metadata

class FullRequest(BaseModel):
    input_image: InputImage
    queries: Queries
    
    