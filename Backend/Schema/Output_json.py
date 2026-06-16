from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


# --------------------------
# INPUT IMAGE SECTION
# --------------------------

class Metadata(BaseModel):
    width: int
    height: int
    spatial_resolution_m: float


class InputImage(BaseModel):
    image_id: str
    image_url: str
    metadata: Metadata


# --------------------------
# CAPTION QUERY
# --------------------------

class CaptionQuery(BaseModel):
    instruction: str
    response: str


# --------------------------
# GROUNDING QUERY
# --------------------------

class GroundingObject(BaseModel):
    object_id: str = Field(alias="object-id")
    obbox: List[int]   # [x1, y1, x2, y2, angle]


class GroundingQuery(BaseModel):
    instruction: str
    response: List[GroundingObject]


# --------------------------
# ATTRIBUTE QUERY
# --------------------------

class BinaryAttribute(BaseModel):
    instruction: str
    response: str


class NumericAttribute(BaseModel):
    instruction: str
    response: float


class SemanticAttribute(BaseModel):
    instruction: str
    response: str


class AttributeQuery(BaseModel):
    binary: BinaryAttribute
    numeric: NumericAttribute
    semantic: SemanticAttribute


# --------------------------
# TOP LEVEL RESPONSE
# --------------------------

class FullResponse(BaseModel):
    input_image: InputImage
    queries: dict




def build_final_response(image_id, image_url, width, height, resolution,
                         caption_query,
                         caption_text,
                         grounding_query,
                         grounding_objects,
                         binary_instruction, binary_response,
                         numeric_instruction, numeric_response,
                         semantic_instruction, semantic_response):

    response = FullResponse(
        input_image={
            "image_id": image_id,
            "image_url": image_url,
            "metadata": {
                "width": width,
                "height": height,
                "spatial_resolution_m": resolution
            }
        },
        queries={
            "caption_query": {
                "instruction": caption_query,
                "response": caption_text
            },
            "grounding_query": {
                "instruction": grounding_query,
                "response": [
                    {
                        "object-id": obj["object_id"],
                        "obbox": obj["obbox"]
                    } for obj in grounding_objects
                ]
            },
            "attribute_query": {
                "binary": {
                    "instruction": binary_instruction,
                    "response": binary_response
                },
                "numeric": {
                    "instruction": numeric_instruction,
                    "response": numeric_response
                },
                "semantic": {
                    "instruction": semantic_instruction,
                    "response": semantic_response
                }
            }
        }
    )

    return response.model_dump()
