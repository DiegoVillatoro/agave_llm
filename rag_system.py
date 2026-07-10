import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

#from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

from config import *
from prompts import *

import json
from collections import Counter


import json
from collections import Counter

import json
from pathlib import Path
from collections import Counter, defaultdict

from enum import Enum
from typing import List
from pydantic import BaseModel, Field
import random

# ============================================================
# ENUMS
# ============================================================

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FieldLevel(str, Enum):
    WEAK = "weak"
    REVIEW = "review"
    HEALTH = "health"

class RecommendationPriority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class HypothesisCategory(str, Enum):
    PESTS = "pests"
    DISEASES = "diseases"  
    NUTRIENT_INBALANCE = "nutrient inbalance"
    WATER_STRESS = "water stress"
    NATURAL_VARIABILITY = "natural variability"
    OTHER = "others"

# ============================================================
# PHENOTYPIC PATTERN
# ============================================================
class PhenotypicPattern(BaseModel):

    name: str = Field(
        description="Name of the observed pattern."
    )

    prevalence_percent: float = Field(
        ge=0,
        le=100,
        description="Percentage of analyzed regions where the pattern was observed."
    )

    strength_score: float = Field(
        ge=0,
        le=1,
        description="Normalized strength score."
    )

# ============================================================
# FIELD ASSESSMENT
# ============================================================

class FieldAssessment(BaseModel):

    condition_field: FieldLevel

    confidence: ConfidenceLevel

    overall_interpretation: str = Field(
        description="High-level agronomic interpretation of field condition."
    )


# ============================================================
# OBSERVED PATTERNS
# ============================================================

class ObservedPatterns(BaseModel):

    color_indicators: List[PhenotypicPattern] = Field(
        default_factory=list,
        description="Observed canopy color-related indicators."
    )

    structural_indicators: List[PhenotypicPattern] = Field(
        default_factory=list,
        description="Observed canopy structure indicators."
    )

    texture_indicators: List[PhenotypicPattern] = Field(
        default_factory=list,
        description="Observed texture indicators."
    )

    summary: str

# ============================================================
# HYPOTHESIS
# ============================================================

class Hypothesis(BaseModel):

    hypothesis: str

    category: HypothesisCategory

    evidence_support_score: int = Field(
        ge=0,
        le=100,
        description="Evidence support score from 0 to 100."
    )

    supporting_observations: List[str] = Field(
        default_factory=list
    )

    supporting_retrieved_evidence: List[str] = Field(
        default_factory=list
    )


# ============================================================
# UNCERTAINTY ANALYSIS
# ============================================================

class UncertaintyAnalysis(BaseModel):

    major_uncertainties: List[str] = Field(
        default_factory=list
    )

    required_additional_information: List[str] = Field(
        default_factory=list
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

class Recommendation(BaseModel):

    priority: RecommendationPriority

    action: str

    justification: str

# ============================================================
# ROOT MODEL
# ============================================================

class AgronomicAnalysis(BaseModel):

    field_assessment: FieldAssessment

    observed_patterns: ObservedPatterns

    hypotheses: List[Hypothesis] = Field(
        default_factory=list
    )

    uncertainty_analysis: UncertaintyAnalysis

    recommendations: List[Recommendation] = Field(
        default_factory=list
    )

    def to_text(self) -> str:

        lines = []

        # =====================================================
        # FIELD ASSESSMENT
        # =====================================================

        lines.append("FIELD ASSESSMENT")
        lines.append("-" * 80)

        lines.append(
            f"Condition field: "
            f"{self.field_assessment.condition_field.value}"
        )

        lines.append(
            f"Confidence: "
            f"{self.field_assessment.confidence.value}"
        )

        lines.append(
            f"Overall interpretation: "
            f"{self.field_assessment.overall_interpretation}"
        )

        lines.append("")

        # =====================================================
        # OBSERVED PATTERNS
        # =====================================================

        lines.append("OBSERVED PATTERNS")
        lines.append("-" * 80)

        if self.observed_patterns.color_indicators:

            lines.append("Color indicators:")

            for item in self.observed_patterns.color_indicators:

                lines.append(
                    f"  • {item.name} "
                    f"(prevalence: {item.prevalence_percent:.1f}%, "
                    f"strength: {item.strength_score:.2f})"
                )

            lines.append("")


        if self.observed_patterns.structural_indicators:

            lines.append("Structural indicators:")

            for item in self.observed_patterns.structural_indicators:

                lines.append(
                    f"  • {item.name} "
                    f"(prevalence: {item.prevalence_percent:.1f}%, "
                    f"strength: {item.strength_score:.2f})"
                )

            lines.append("")


        if self.observed_patterns.texture_indicators:

            lines.append("Texture indicators:")

            for item in self.observed_patterns.texture_indicators:

                lines.append(
                    f"  • {item.name} "
                    f"(prevalence: {item.prevalence_percent:.1f}%, "
                    f"strength: {item.strength_score:.2f})"
                )

            lines.append("")
        lines.append("")
        lines.append(self.observed_patterns.summary)
        lines.append("")

        # =====================================================
        # HYPOTHESES
        # =====================================================

        lines.append("PLAUSIBLE EXPLANATIONS")
        lines.append("-" * 80)

        ranked = sorted(
            self.hypotheses,
            key=lambda x: x.evidence_support_score,
            reverse=True
        )

        for idx, hypothesis in enumerate(ranked, start=1):

            lines.append(
                f"{idx}. {hypothesis.hypothesis}"
            )

            lines.append(
                f"   Category: "
                f"{hypothesis.category.value}"
            )

            lines.append(
                f"   Evidence support score: "
                f"{hypothesis.evidence_support_score}/100"
            )

            if hypothesis.supporting_observations:
                lines.append(
                    "   Supporting observations:"
                )

                for obs in hypothesis.supporting_observations:
                    lines.append(f"      - {obs}")

            if hypothesis.supporting_retrieved_evidence:
                lines.append(
                    "   Supporting retrieved evidence:"
                )

                for ret in hypothesis.supporting_retrieved_evidence:
                    lines.append(f"      - {ret}")

            lines.append("")

        # =====================================================
        # UNCERTAINTY
        # =====================================================

        lines.append("UNCERTAINTY ANALYSIS")
        lines.append("-" * 80)

        for item in self.uncertainty_analysis.major_uncertainties:
            lines.append(f"• {item}")

        lines.append("")

        lines.append(
            "Additional information required:"
        )

        for item in (
            self.uncertainty_analysis
            .required_additional_information
        ):
            lines.append(f"• {item}")

        lines.append("")

        # =====================================================
        # RECOMMENDATIONS
        # =====================================================

        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)

        priority_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        recommendations = sorted(
            self.recommendations,
            key=lambda r:
            priority_order.get(
                r.priority.value,
                99
            )
        )

        for rec in recommendations:

            lines.append(
                f"[{rec.priority.value}] "
                f"{rec.action}"
            )

            lines.append(
                f"Reason: {rec.justification}"
            )

            lines.append("")

        return "\n".join(lines)

def load_tiles(folder):

    tiles = []

    for file in Path(folder).glob("*.json"):

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            tiles.extend(data)

    return tiles

def compute_tile_score(tile):

    summary = tile["analysis"]["regional_summary"]

    anomaly_score = summary.get(
        "score_anomaly_condition",
        0,
    )
    similarity_expected = summary.get(
        "score_similarity_expected",
        0,
    )

    return anomaly_score / ( anomaly_score + similarity_expected)

def extract_tiles(
    tiles,
    tiles_to_analyze
):
    """
    Keep tiles according to tiles_to_analyze.
    """
    ranked_tiles = sorted(
        tiles,
        key=compute_tile_score,
        reverse=True, #larger to smaller
    )

    selected = []

    for tile in ranked_tiles:

        summary = tile["analysis"]["regional_summary"]
        #print(summary.get("score_anomaly_condition", 0))
        #anomaly_score = summary.get(
        #    "score_anomaly_condition",
        #    0,
        #)

        condition = summary.get(
            "condition_classification",
            "",
        )

        if (
            #condition != "Consistent with Expected Condition"
            #condition == "Anomalous Condition"
            #condition == "Intermediate / Uncertain"
            condition == tiles_to_analyze or tiles_to_analyze=="ALL"
        ):
            selected.append(tile)

    return selected

def collect_features(tiles):
    MAX_STRENGTH = 5.0
    total_tiles = len(tiles)

    feature_strength_sum = defaultdict(float)
    feature_tile_count = defaultdict(int)

    overall_reasonings = []

    for tile in tiles:

        present_features = set() #per tile

        for obs in tile["analysis"]["observations"]:

            strength = (
                obs["feature_strength_score"]
                / MAX_STRENGTH
            ) #normalize score 0-1

            features = (
                obs.get("canopy_color_features", [])
                + obs.get("structural_features", [])
                + obs.get("texture_features", [])
            )

            for feature in features:

                feature_strength_sum[feature] += strength

                if feature not in present_features:
                    feature_tile_count[feature] += 1 #only 1 count for that feature per tile
                    present_features.add(feature)

        overall_reasonings.append(
            tile["analysis"]["regional_summary"]
            ["overall_reasoning"]
        )

    feature_scores = {}

    for feature in feature_strength_sum:

        count = feature_tile_count[feature] #count of how many tiles present that feature

        mean_strength = (
            feature_strength_sum[feature] / count
        )

        prevalence = count / total_tiles

        feature_scores[feature] = (
            mean_strength,
            prevalence
        )

    return feature_scores, overall_reasonings

def build_retrieval_query(
    filter_tiles,
    tiles_to_analyze,
    top_k=10,
):
    
    feature_scores, overall_reasonings = collect_features(
        filter_tiles
    )

    ranked = sorted(
        feature_scores.items(),
        key=lambda x: x[1][1],
        reverse=True,
    )
    top_features = ranked[:top_k]

    query = []

    query.append(
        f"Agave field summary for analysis from "
        f"{len(filter_tiles)} {tiles_to_analyze} regions."
    )

    query.append("")
    query.append(
        "Dominant phenotypic symptoms indicating percentage of affected regions and "
        "the average streng score from 0 (not observed) to 1 (very strong evidence):"
    )

    for feature, score in top_features:

        query.append(
            f"- {feature} (prevalence on {int(score[1]*100)}% of the analyzed regions with a strength score: {score[0]:.1f})"
        )

    query.append("")
    query.append(
        "Reasonings samples associated with these symptoms. It correspond to random samples taken from "+tiles_to_analyze+" regions:"
    )
    random_reasonings = random.sample(overall_reasonings, min(top_k, len(overall_reasonings))) #select top_k reasonings from tiles corresponding to "tiles_to_analyze"

    for reasoning in random_reasonings:
        query.append(
            f" - {reasoning}"
        )

    query.append("")
    query.append(
        "Provide key explanation linking "
        "observed canopy color, canopy structure, and "
        "texture distribution patterns."
    )
    query.append("\n")
    return "\n".join(query)

def build_orthomap_query(descriptions_dir, tiles_to_analyze):
    
    #load all tiles from directory
    print(descriptions_dir)
    tiles = load_tiles(descriptions_dir)
    print("*************load tiles", len(tiles))
    interest_tiles = extract_tiles(
        tiles,
        tiles_to_analyze
    )
    #print("Extracted anomaous tiles")

    query = build_retrieval_query(
        interest_tiles,
        tiles_to_analyze
    )
    return query

class VectorRAGSystem:
    """
    MULTIMODAL RAG SYSTEM

    Features:
    - Text retrieval (OpenAI embeddings)
    - Image retrieval (CLIP embeddings)
    - Image query support
    - Hybrid retrieval
    - GPT-4o image understanding
    """

    def __init__(
        self,
        chroma_text_path: str = "./chroma_text",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #######################################################################
        # PATHS
        #######################################################################

        self.chroma_text_path = Path(chroma_text_path)

        #######################################################################
        # TEXT EMBEDDINGS
        #######################################################################

        print("🔤 Loading OpenAI embeddings...")

        self.text_embeddings = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL
        )

        #######################################################################
        # LLM
        #######################################################################

        print("🤖 Loading GPT model...")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        #######################################################################
        # VECTORSTORES
        #######################################################################

        self.text_vectorstore = None

        #######################################################################
        # LOGGING
        #######################################################################

        logging.basicConfig()

        logging.getLogger(
            "langchain.retrievers.multi_query"
        ).setLevel(logging.INFO)

        #######################################################################
        # LOAD DBs
        #######################################################################

        self._load_vectorstores()

    ###########################################################################
    # LOAD VECTORSTORES
    ###########################################################################

    def _load_vectorstores(self):

        try:

            ###################################################################
            # TEXT VECTORSTORE
            ###################################################################

            if self.chroma_text_path.exists():

                self.text_vectorstore = Chroma(
                    persist_directory=str(
                        self.chroma_text_path
                    ),
                    embedding_function=self.text_embeddings,
                    collection_name="disease_text_knowledge"
                )

                self.text_retriever = (
                    MultiQueryRetriever.from_llm(
                        retriever=self.text_vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5}
                        ),
                        llm=self.llm,
                        prompt=self._get_multi_query_prompt()
                    )
                )

                print("✅ Text vectorstore loaded")

        except Exception as e:

            print(
                f"❌ Error loading vectorstores: {e}"
            )

    ###########################################################################
    # MULTI QUERY PROMPT
    ###########################################################################

    def _get_multi_query_prompt(self):

        return ChatPromptTemplate.from_template(
            PROMPT_RETRIEVER
        )

    ###########################################################################
    # TEXT SEARCH
    ###########################################################################

    def search_text(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:

        if not self.text_retriever:

            return []

        docs = self.text_retriever.invoke(query)

        return docs[:k]

    ###########################################################################
    # HYBRID MULTIMODAL SEARCH
    ###########################################################################

    def search_query_description(
        self,
        descriptions_dir: str,
        tiles_to_analyze: str
    ) -> Dict[str, Any]:
        """
        Full multimodal search from image input.
        """

        try:

            ###################################################################
            # STEP 1
            # FIELD UNDERSTANDING
            ###################################################################

            print(
                "🧠 Generating query based on descriptions files..."
            )
            #descriptions_dir = "E:\\Experiments\\agents\\agave\\map_Zone102_part1\\Descriptions"
            generated_query = build_orthomap_query(descriptions_dir, tiles_to_analyze)
            
            ###################################################################
            # STEP 2
            # TEXT RETRIEVAL
            ###################################################################

            print(
                "📚 Searching text knowledge..."
            )

            text_docs = self.search_text(
                generated_query,
                k=5
            )
            #print(text_docs)

            ###################################################################
            # STEP 3
            # BUILD CONTEXT
            ###################################################################

            context_parts = []
            sources = set()

            ###############################################################
            # TEXT DOCUMENTS
            ###############################################################

            if text_docs:
                context_parts.append("=== EVIDENCE ===")

                for i, doc in enumerate(text_docs):
                    content = doc.page_content.strip()

                    if not content:
                        continue

                    filename = doc.metadata.get("filename", f"text_doc_{i+1}")
                    score = doc.metadata.get("score", None)
                    page = doc.metadata.get("page", None)

                    section = [
                        f"TEXT DOCUMENT {i+1}",
                        f"Source: {filename}"
                    ]

                    if page is not None:
                        section.append(f"Page: {page}")

                    if score is not None:
                        section.append(f"Similarity Score: {score:.4f}")

                    section.append("Content:")
                    section.append(content)

                    context_parts.append("\n".join(section))
                    sources.add(filename)

            ###############################################################
            # FINAL CONTEXT
            ###############################################################

            context = "\n\n" + ("\n" + "=" * 80 + "\n\n").join(context_parts)
            

            ###################################################################
            # STEP 4
            # CALL STRUCTURED LLM
            ###################################################################

            final_prompt = ChatPromptTemplate.from_template(
                MAIN_PROMPT
            )
            structured_llm = self.llm.with_structured_output(
                AgronomicAnalysis
            )
            
            response = structured_llm.invoke(
                final_prompt.format(
                    query=generated_query,
                    context=context
                )
            )
            print("✅ llm response")

            ###################################################################
            # STEP 5
            # SAVE RESULTS
            ###################################################################
            from pathlib import Path
            file_path = (Path(descriptions_dir).parent / f"OVERALL_{tiles_to_analyze.replace("/", "-")}.json")
            data = response.model_dump()
            data["generated_query"] = generated_query
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            print("✅ Saved llm response at: "+ str(file_path))

            return {
                "generated_query": generated_query,
                "respuesta": response.to_text(),
                #"confianza": confidence,
                "text_documents": len(text_docs),
                "fuentes": list(sources)
            }

        except Exception as e:

            return {
                "respuesta": f"Error: {e}",
                #"confianza": 0.0,
                "fuentes": []
            }