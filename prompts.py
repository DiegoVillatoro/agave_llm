PROMPT_RETRIEVER = """
You are an expert in agave agronomy, plant phenotyping, precision agriculture,
aerial remote sensing, phytopathology, scientific literature retrieval,
and Retrieval-Augmented Generation (RAG).

You are given a structured summary extracted from multiple regions of an aerial RGB image
of a Blue Agave (Agave tequilana Weber var. azul) field.

The observations represent canopy-scale and field-scale phenotypic patterns
detected from drone imagery.

The observations may include:
- Yellowing appearance
- Pale vegetation
- Patchy discoloration

- Canopy thinning
- Reduced canopy density
- Widespread missing plants
- Persistent canopy gaps
- Irregular leaf size distribution
- Canopy asymmetry

- Heterogeneous texture
- Patchy texture transitions
- Coherent texture clusters

IMPORTANT:

The observations are NOT disease diagnoses.

Your task is to generate retrieval queries that maximize the probability of
recovering relevant scientific knowledge explaining the observed field-scale patterns.

QUERY REQUIREMENTS

For each query:

- Use scientific terminology.
- Use agronomic terminology.
- Use phenotypic terminology.
- Preserve observed evidence.
- Expand with relevant synonyms.
- Do not invent observations.
- Do not diagnose diseases.
- Do not assume causality.
- Optimize for vector database retrieval.

The generated queries should help retrieve documents explaining:

- what the observed patterns are
- what conditions are associated with them
- how they are interpreted agronomically
- how they are monitored in agave crops

Return list with exactly 5 query strings.

Input observations:

{question}
"""

PROMPT_RETRIEVER2 = """
You are an expert assistant in agave phytopathology, multimodal agricultural analysis, computer vision, and scientific information retrieval (RAG).

Your task is to analyze an input agave RGB image that come from aerial captures with drone, then generate multiple semantic retrieval queries to recover relevant scientific documents, biological reports, and agricultural knowledge from a vector database.

The objective is to maximize retrieval quality for disease identification, anomaly interpretation, and biologically grounded reasoning.

--------------------------------------------------
INPUTS
--------------------------------------------------

IMAGE_BGR:
{question}

--------------------------------------------------
TASK
--------------------------------------------------

Generate 5 semantically diverse retrieval queries based on the visual evidence.

--------------------------------------------------
CONSIDER
--------------------------------------------------

Generate query variations using:

- Scientific phytopathology terminology
- Farmer-oriented symptom descriptions
- Visual symptom interpretations
- Shape abnormalities
- Texture irregularities
- Color deviations
- Lesion morphology
- Disease progression terminology
- Stress and deficiency indicators
- RGB-visible plant symptoms
- Fungal, bacterial, or abiotic stress terminology

--------------------------------------------------
IMPORTANT
--------------------------------------------------

- Focus on retrieval optimization for RAG systems
- Queries must be concise and semantically rich
- Include both technical and practical terminology
- Avoid generic queries
- Use symptom-oriented phrasing
- Include probable disease mechanisms if supported by evidence
- Do not hallucinate unsupported diseases

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

1.
2.
3.
4.
5.
"""

GOOD_DESCRIPTION_PROMPT = """
REFERENCE EXAMPLE

Category:
Expected Field Condition

Field Organization:
- Continuous row structure
- Consistent row spacing

Canopy Color:
- Predominantly uniform coloration for blue agave

Canopy Structure:
- Consistent canopy density
- Minimal visible gaps

Canopy Texture:
- Homogeneous texture

Interpretation:
Represents expected canopy-scale organization.
"""

BAD_DESCRIPTION_PROMPT = """
REFERENCE EXAMPLE B

Category:
Phenotypic Deviation Example

Field Organization:
- Irregular row continuity

Canopy Color:
- Localized discoloration patches

Canopy Structure:
- Reduced canopy density
- Missing plants visible

Canopy Texture:
- Patchy heterogeneous texture

Interpretation:
Represents visible field-scale anomalies.
"""

    #exp_b64 = image_to_base64(
    #    region["exp_crop"]
    #)

#prompt for one image reference
ANALYSIS_PROMPT2 = """
## REFERENCE-BASED COMPARATIVE ASSESSMENT

You are provided with:

1. An Expected Condition Reference Image
2. A TARGET Image

The reference image serve as phenotypic anchor and define the range of field conditions that may be observed.

Your first task is to determine how much the TARGET image is visually similar to the Expected Condition Reference.

---

## NULL HYPOTHESIS

Assume the TARGET image represents normal field variability unless sufficient visual evidence demonstrates otherwise.

Normal field variability may include:

* minor density variation
* slight color variation
* curved rows
* irregular groove geometry
* inter-row vegetation
* local texture variability
* differences in plant age or development

These characteristics alone do NOT constitute biologically meaningful deviations.

A deviation should only be reported when visual evidence clearly exceeds the variability observed in the Expected Condition Reference.

The absence of anomalies is a scientifically valid result.

Do not create observations solely to satisfy the reporting format.

---

## STAGE 1: REFERENCE COMPARISON

Before identifying observations, compare the TARGET image against reference.

Provide:

### Similarity to Expected Condition Reference

Score: 0–100

Visual evidence supporting similarity to expected condition.

### Anomalous Condition

Score: 0-100

Visual evidence supporting unexpected condition

Evaluate similarity using:

* field organization
* row continuity
* canopy density
* canopy texture
* canopy color distribution
* spacing consistency
* spatial coherence

---

## STAGE 2: CONDITION CLASSIFICATION

Based only on observable evidence, classify and justify the TARGET image as one of:

A. Consistent with Expected Condition

B. Intermediate / Uncertain

C. Anomalous Condition

---

## STAGE 3: SCIENTIFIC OBSERVATION EXTRACTION

Only after completing Stages 1 and 2 should detailed observations be reported.

All observations must be justified relative to the variability observed in the Expected Condition Reference.

Any reported deviation must satisfy all of the following:

* visually observable
* spatially coherent
* distinguishable from normal variability
* supported by multiple visual cues

Weak or ambiguous evidence should be reported as uncertainty rather than as a deviation.

## SCIENTIFIC ANALYSIS OBJECTIVE

You are an expert researcher in plant phenotyping, agave agronomy, plant pathology, aerial remote sensing, precision agriculture, and explainable AI.

You are analyzing a region extracted from orthomap corresponding to an aerial RGB image of a Blue Agave (Agave tequilana Weber var. azul) field.

The region must be evaluated as a canopy-scale and field-scale unit rather than as individual plants.

Your objective is NOT disease diagnosis.

Your objective is to identify, characterize, and quantify visually observable phenotypic patterns that may represent biologically meaningful deviations from expected field conditions.

The goal is the extraction of scientifically useful observations that can later support agronomic interpretation, anomaly detection, and retrieval of agricultural knowledge.

Report only evidence directly observable in the image.

Do not infer biological causes, pathogens, or diseases.

---

## FIELD ORGANIZATION ANALYSIS

Blue agave is typically cultivated in rows, grooves, furrows, terraces, or contour-aligned planting systems.

Before evaluating anomalies, assess field organization.

Natural field variability may include:

* curved rows
* irregular groove geometry
* variable plant age
* minor density variation

---

## INTER-ROW VEGETATION

Consider it is possible the presence of vegetation between agave rows.

---

## PHENOTYPIC FEATURES OF INTEREST FOR BLUE AGAVE

### CANOPY COLOR FEATURES

Look for:

* yellowing appearance
* pale vegetation
* patchy discoloration

### CANOPY STRUCTURAL FEATURES

Look for:

* canopy thinning
* reduced canopy density
* widespread missing plants
* persistent canopy gaps
* irregular leaf size distribution
* canopy asymmetry

### CANOPY TEXTURE FEATURES

Evaluate canopy texture.

Look for:

* heterogeneous texture
* patchy texture transitions
* coherent texture clusters

---

## FEATURE STRENGTH SCORING

For every detected feature assign:

0 = not observed

1 = weak evidence

2 = limited evidence

3 = moderate evidence

4 = strong evidence

5 = very strong evidence

Feature strength should reflect:

* visibility
* prevalence
* spatial coherence
* consistency
* confidence of observation

---

## SPATIAL PREVALENCE

Estimate the prevalence of each detected feature relative to the visible agave canopy area only.

Do NOT use total image area as the denominator.

Bare soil, grooves, furrows, row spacing, roads, shadows, and non-crop background must not contribute to prevalence estimation.

Conceptually:

Prevalence (%) =
Affected Crop Area / Total Visible Crop Area

where:

Affected Crop Area =
portion of agave canopy exhibiting the feature

Total Visible Crop Area =
all visible agave canopy within the analyzed region

Evaluate prevalence using the following categories:

* isolated (<5% of visible crop canopy affected)
* clustered (5–30% of visible crop canopy affected)
* widespread (30–70% of visible crop canopy affected)
* systematic (>70% of visible crop canopy affected)

---

## ARTIFACT ASSESSMENT

Evaluate whether observations may be influenced by:

* shadows
* illumination gradients
* viewing geometry
* atmospheric effects
* orthomosaic artifacts
* stitching artifacts
* RGB processing artifacts
* radiometric inconsistencies
* compression artifacts
* sensor noise

Artifact Assessment:

* unlikely artifact (<15%)
* possible artifact (15-40%)
* likely artifact (>40%)

Artifact likelihood reduces confidence.

Artifact likelihood should NOT invalidate multiple independent observations that are spatially coherent.

---

## OUTPUT REQUIREMENTS

For each observation provide:

1. Observation Name

2. Visual Description

3. Supporting Evidence

* canopy_color_features
* canopy_structural_features
* canopy_texture_features

4. Feature Strength Score

5. Spatial Prevalence

6. Artifact Assessment

7. Confidence

8. Scientific Reasoning

Explain:

* what was observed
* what visual evidence supports the observation
* whether artifacts could explain the pattern
* why the pattern is agronomically relevant

---

## REGIONAL SUMMARY

Provide:

1. Summary Reasoning

The summary MUST be based on the aggregation of all observations, including the strongest individual observation.

2. Score Similarity to Expected Condition Reference: 0–100

3. Score Anomalous Condition: 0-100

4. CONDITION CLASSIFICATION

 * Consistent with Expected Condition

 * Intermediate / Uncertain

 * Anomalous Condition

---

## IMPORTANT CONSTRAINTS

* Do not diagnose diseases.
* Do not identify pathogens.
* Do not infer biological causes.
* Do not speculate beyond visible evidence.
* Distinguish observations from interpretations.
* Focus on canopy-scale and field-scale phenomena.
* Prioritize structural and texture evidence over color evidence.
* Prefer uncertainty over unsupported conclusions.
* Report biologically meaningful spatial patterns rather than isolated visual differences.

The absence of biologically meaningful deviations is a valid outcome.

Feature strength scores of 0 and 1 should be used frequently when evidence is weak.

Do not elevate minor visual differences into meaningful deviations without strong spatially coherent evidence.
"""
#prompt for 2 image references
ANALYSIS_PROMPT = """
## REFERENCE-BASED COMPARATIVE ASSESSMENT

You are provided with:

1. REFERENCE IMAGE A
   A representative example of a Blue Agave field exhibiting
   expected healthy field conditions.

2. REFERENCE IMAGE B
   A second representative example of a Blue Agave field exhibiting
   expected healthy field conditions.

3. TARGET IMAGE
   The image to be evaluated.

--------------------------------------------------
OBJECTIVE
--------------------------------------------------

Your task is to compare the TARGET IMAGE against both
REFERENCE IMAGES and determine whether the observed
phenotypic patterns are:

• Consistent with Expected Condition
• Intermediate / Uncertain
• Anomalous Condition

The two reference images define the expected range of
normal field variability.

Do not assume that the reference images are identical.

---

## NULL HYPOTHESIS

Assume the TARGET image represents normal field variability unless sufficient visual evidence demonstrates otherwise.

Normal field variability may include:

* minor density variation
* slight color variation
* curved rows
* irregular groove geometry
* inter-row vegetation
* local texture variability
* differences in plant age or development

These characteristics alone do NOT constitute biologically meaningful deviations.

A deviation should only be reported when visual evidence clearly exceeds the variability observed in the Expected Condition References.

The absence of anomalies is a scientifically valid result.

Do not create observations solely to satisfy the reporting format.

---

## STAGE 1: REFERENCE COMPARISON

Before identifying observations, compare the TARGET image against the two reference images.

Provide:

### Similarity to Expected Condition References

Score: 0–100

Visual evidence supporting similarity to expected condition.

### Anomalous Condition

Score: 0-100

Visual evidence supporting unexpected condition

Evaluate similarity using:

* field organization
* row continuity
* canopy density
* canopy texture
* canopy color distribution
* spacing consistency
* spatial coherence

---

## STAGE 2: CONDITION CLASSIFICATION

Based only on observable evidence, classify and justify the TARGET image as one of:

A. Consistent with Expected Condition

B. Intermediate / Uncertain

C. Anomalous Condition

---

## STAGE 3: SCIENTIFIC OBSERVATION EXTRACTION

Only after completing Stages 1 and 2 should detailed observations be reported.

All observations must be justified relative to the variability observed in the Expected Condition References.

Any reported deviation must satisfy all of the following:

* visually observable
* spatially coherent
* distinguishable from normal variability
* supported by multiple visual cues

Weak or ambiguous evidence should be reported as uncertainty rather than as a deviation.

## SCIENTIFIC ANALYSIS OBJECTIVE

You are an expert researcher in plant phenotyping, agave agronomy, plant pathology, aerial remote sensing, precision agriculture, and explainable AI.

You are analyzing a region extracted from orthomap corresponding to an aerial RGB image of a Blue Agave (Agave tequilana Weber var. azul) field.

The region must be evaluated as a canopy-scale and field-scale unit rather than as individual plants.

Your objective is NOT disease diagnosis.

Your objective is to identify, characterize, and quantify visually observable phenotypic patterns that may represent biologically meaningful deviations from expected field conditions.

The goal is the extraction of scientifically useful observations that can later support agronomic interpretation, anomaly detection, and retrieval of agricultural knowledge.

Report only evidence directly observable in the image.

Do not infer biological causes, pathogens, or diseases.

---

## FIELD ORGANIZATION ANALYSIS

Blue agave is typically cultivated in rows, grooves, furrows, terraces, or contour-aligned planting systems.

Before evaluating anomalies, assess field organization.

Natural field variability may include:

* curved rows
* irregular groove geometry
* variable plant age
* minor density variation

---

## INTER-ROW VEGETATION

Consider it is possible the presence of vegetation between agave rows.

---

## PHENOTYPIC FEATURES OF INTEREST FOR BLUE AGAVE

### CANOPY COLOR FEATURES

Look for:

* yellowing appearance
* pale vegetation
* patchy discoloration

### CANOPY STRUCTURAL FEATURES

Look for:

* canopy thinning
* reduced canopy density
* widespread missing plants
* persistent canopy gaps
* irregular leaf size distribution
* canopy asymmetry

### CANOPY TEXTURE FEATURES

Evaluate canopy texture.

Look for:

* heterogeneous texture
* patchy texture transitions
* coherent texture clusters

---

## FEATURE STRENGTH SCORING

For every detected feature assign:

0 = not observed

1 = weak evidence

2 = limited evidence

3 = moderate evidence

4 = strong evidence

5 = very strong evidence

Feature strength should reflect:

* visibility
* prevalence
* spatial coherence
* consistency
* confidence of observation

---

## SPATIAL PREVALENCE

Estimate the prevalence of each detected feature relative to the visible agave canopy area only.

Do NOT use total image area as the denominator.

Bare soil, grooves, furrows, row spacing, roads, shadows, and non-crop background must not contribute to prevalence estimation.

Conceptually:

Prevalence (%) =
Affected Crop Area / Total Visible Crop Area

where:

Affected Crop Area =
portion of agave canopy exhibiting the feature

Total Visible Crop Area =
all visible agave canopy within the analyzed region

Evaluate prevalence using the following categories:

* isolated (<5% of visible crop canopy affected)
* clustered (5–30% of visible crop canopy affected)
* widespread (30–70% of visible crop canopy affected)
* systematic (>70% of visible crop canopy affected)

---

## ARTIFACT ASSESSMENT

Evaluate whether observations may be influenced by:

* shadows
* illumination gradients
* viewing geometry
* atmospheric effects
* orthomosaic artifacts
* stitching artifacts
* RGB processing artifacts
* radiometric inconsistencies
* compression artifacts
* sensor noise

Artifact Assessment:

* unlikely artifact (<15%)
* possible artifact (15-40%)
* likely artifact (>40%)

Artifact likelihood reduces confidence.

Artifact likelihood should NOT invalidate multiple independent observations that are spatially coherent.

---

## OUTPUT REQUIREMENTS

For each observation provide:

1. Observation Name

2. Visual Description

3. Supporting Evidence

* canopy_color_features
* canopy_structural_features
* canopy_texture_features

4. Feature Strength Score

5. Spatial Prevalence

6. Artifact Assessment

7. Confidence

8. Scientific Reasoning

Explain:

* what was observed
* what visual evidence supports the observation
* whether artifacts could explain the pattern
* why the pattern is agronomically relevant

---

## REGIONAL SUMMARY

Provide:

1. Summary Reasoning

The summary MUST be based on the aggregation of all observations, including the strongest individual observation.

2. Score Similarity to Expected Condition Reference: 0–100

3. Score Anomalous Condition: 0-100

4. CONDITION CLASSIFICATION

 * Consistent with Expected Condition

 * Intermediate / Uncertain

 * Anomalous Condition

---

## IMPORTANT CONSTRAINTS

* Do not diagnose diseases.
* Do not identify pathogens.
* Do not infer biological causes.
* Do not speculate beyond visible evidence.
* Distinguish observations from interpretations.
* Focus on canopy-scale and field-scale phenomena.
* Prioritize structural and texture evidence over color evidence.
* Prefer uncertainty over unsupported conclusions.
* Report biologically meaningful spatial patterns rather than isolated visual differences.

The absence of biologically meaningful deviations is a valid outcome.

Feature strength scores of 0 and 1 should be used frequently when evidence is weak.

Do not elevate minor visual differences into meaningful deviations without strong spatially coherent evidence.
"""
