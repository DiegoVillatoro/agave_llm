#### EXPLAIN TILES
import os
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from openai import OpenAI
from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from prompts import *

client = OpenAI(
    #api_key=OPENAI_API_KEY
)

# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

# --------------------------------------------------
# ENUMS
# --------------------------------------------------

class Confidence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SpatialPrevalence(str, Enum):
    ISOLATED = "Isolated (<5%)"
    CLUSTERED = "Clustered (5-30%)"
    WIDESPREAD = "Widespread (30-70%)"
    SYSTEMATIC = "Systematic (>70%)"


class ArtifactAssessment(str, Enum):
    UNLIKELY_ARTIFACT = "Unlikely artifact (<15%)"
    POSSIBLE_ARTIFACT = "Possible artifact (15-40%)"
    LIKELY_ARTIFACT = "Likely artifact (>40%)"


class AgronomicRelevance(str, Enum):
    HEALTHY = "Healthy"
    MOSTLY_HEALTHY = "Mostly healthy"
    MODERATELY_STRESSED = "Moderately stressed"
    HIGHLY_STRESSED = "Highly stressed"


# --------------------------------------------------
# PHENOTYPIC FEATURES
# --------------------------------------------------

class CanopyColorFeature(str, Enum):
    CHLOROTIC_APPEARANCE = "Yellowing appearance"
    PALE_VEGETATION = "Pale vegetation"
    PATCHY_DISCOLORATION = "Patchy discoloration"

class StructuralFeature(str, Enum):
    CANOPY_THINNING = "Canopy thinning"
    REDUCED_CANOPY_DENSITY = "Reduced canopy density"
    WIDESPREAD_MISSING_PLANTS = "Widespread missing plants"
    PERSISTENT_CANOPY_GAPS = "Persistent canopy gaps"
    IRREGULAR_LEAF_DISTRIBUTION = "Irregular leaf size distribution"
    CANOPY_ASYMMETRY = "Canopy asymmetry"

class TextureFeature(str, Enum):
    HETEROGENEOUS_TEXTURE = "Heterogeneous texture"
    PATCHY_TEXTURE_TRANSITIONS = "Patchy texture transitions"
    COHERENT_TEXTURE_CLUSTERS = "Coherent texture clusters"

#class SpatialFeature(str, Enum):
#    CLUSTERED_ANOMALIES = "Clustered anomalies"
#    CONTIGUOUS_ANOMALOUS_ZONES = "Contiguous anomalous zones"
#    PATCHY_DISTRIBUTION = "Patchy distribution"
#    REPEATED_LOCALIZED_ANOMALIES = "Repeated localized anomalies"

class ConditionClassification(str, Enum):
    CONSISTENT_EXPECTED = "Consistent with Expected Condition"
    INTERMEDIATE = "Intermediate / Uncertain"
    ANOMALOUS_CONDITION = "Anomalous Condition"
# --------------------------------------------------
# OBSERVATION
# --------------------------------------------------

class Observation(BaseModel):
    observation_name: str = Field(
        description="Short name summarizing the detected observation."
    )
    #2
    visual_description: str = Field(
        description="Objective description of the observed visual pattern."
    )
    #3

    canopy_color_features: List[CanopyColorFeature] = Field(
        default_factory=list
    )

    structural_features: List[StructuralFeature] = Field(
        default_factory=list
    )

    texture_features: List[TextureFeature] = Field(
        default_factory=list
    )

    #4
    feature_strength_score: int = Field(ge=0, le=5)
    #5
    spatial_prevalence: SpatialPrevalence

    artifact_assessment: ArtifactAssessment

    confidence: Confidence

    scientific_reasoning: str = Field(
        description="Explanation of the visual evidence supporting the observation."
    )

# --------------------------------------------------
# OVERALL REGION SUMMARY
# --------------------------------------------------

class RegionalSummary(BaseModel):
    #dominant_canopy_condition: str = Field(
    #    description="Overall canopy condition observed in the region."
    #)

    #dominant_spatial_pattern: str = Field(
    #    description="Main spatial organization pattern."
    #)

    #agronomic_relevance: AgronomicRelevance

    #confidence: Confidence

    overall_reasoning: str = Field(
        description="Summary of the visual evidence supporting the observations."
    )
    score_similarity_expected: int = Field(ge=0, le=100)
    score_anomaly_condition: int = Field(ge=0, le=100)

    #summary_reasoning: str
    condition_classification: ConditionClassification

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

class RegionAnalysis(BaseModel):
    observations: List[Observation] = Field(
        default_factory=list,
        description="List of significant phenotypic observations."
    )

    regional_summary: RegionalSummary

def image_to_base64(path):

    with open(path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


# --------------------------------------------------
# LLM ANALYSIS
# --------------------------------------------------

def analyze_region_with_llm(rgb_path, good_img_path1, good_img_path2):

    good_b64_1 = image_to_base64(good_img_path1)
    good_b64_2 = image_to_base64(good_img_path2)
    #bad_b64 = image_to_base64(bad_img_path)
    target_b64 = image_to_base64(
        rgb_path
    )
    print("llm calling ...")
    response = client.responses.parse(
        #model="gpt-5.6-sol",
        model="gpt-4o-mini",
        temperature=0.0,
        input=[
            {
                "role": "user",
                "content": [

                    {
                        "type": "input_text",
                        "text": GOOD_DESCRIPTION_PROMPT
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{good_b64_1}"
                    },

                    {
                        "type": "input_text",
                        "text": GOOD_DESCRIPTION_PROMPT
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{good_b64_2}"
                    },

                    {
                        "type": "input_text",
                        "text": "TARGET IMAGE"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{target_b64}"
                    },

                    {
                        "type": "input_text",
                        "text": ANALYSIS_PROMPT
                    }
                ]
            }
        ],
        text_format=RegionAnalysis
    )
    print("llm response obtained")
    return response.output_parsed.model_dump()


# --------------------------------------------------
# MAIN
import random
import glob
def create_descriptions(
        directory_path,
        good_samples_path
):
    counter = 0
    good_samples_files = glob.glob(str(good_samples_path)+'/*.png')
    if len(good_samples_files)<2:
        print(f"Good samples: {len(good_samples_files)}. Not enough for analyze")
    else:
        good_samples = random.sample(
                good_samples_files,
                min(
                    2,
                    len(good_samples_files)
                )
            )
        good_img_path1 = good_samples[0]
        good_img_path2 = good_samples[1]
        
        JSONS_PATH = (
                    directory_path / f"Descriptions"
                )
        JSONS_PATH.mkdir(parents=True, exist_ok=True)
            
        # Loop through all files ending with '.txt'
        for RGB_IMAGE_PATH in directory_path.glob("*.png"):
            RGB_IMAGE = str(RGB_IMAGE_PATH)
            RGB_IMAGE_NAME = RGB_IMAGE.split("\\")[-1].split(".")[0]

            JSON_FILE = "field_report_"+RGB_IMAGE_NAME+".json"
            JSON_PATH = (JSONS_PATH / JSON_FILE)

            _, id_y, id_x = RGB_IMAGE_NAME.split("_")
            
            if JSON_PATH.is_file(): #for no call llm if description exist
                print("It was found description for this tile")
                continue
            else:
                analysis = analyze_region_with_llm(RGB_IMAGE, good_img_path1, good_img_path2)
                results = []
                results.append({
                    "region_id": counter,
                    "x": int(id_x),
                    "y": int(id_y),
                    "analysis": analysis
                })

                with open(
                    JSON_PATH,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        results,
                        f,
                        indent=4,
                        ensure_ascii=False
                    )

                print(
                    "Finished. Report saved to "+
                    JSON_FILE
                )
            counter+=1
    return counter

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
#zone_name = "Zone102_part1" 
#zone_name = "Zone108_octubre"
#directory_path = Path("E:\\Experiments\\agents\\agave\\map_"+zone_name)
#good_samples = Path(directory_path, "healthy_samples")

#counter = create_descriptions(
#        directory_path,
#        good_samples
#        )