# HTP Drawing Analyzer
## Ancient Wisdom Meets Modern AI

HTP Drawing Analyzer is an AI-powered system designed to analyze
House-Tree-Person (HTP) drawings and generate structured psychological
interpretations. The project integrates object detection, vision-based
reasoning, retrieval-augmented generation (RAG), and prompt engineering
to simulate the analytical process of an HTP psychologist.

This repository presents two complete approaches, demonstrating the
evolution from a detection-centric pipeline to a hybrid vision-intelligent
system.

--------------------------------------------------
PROJECT OBJECTIVES
--------------------------------------------------
- Analyze hand-drawn HTP sketches
- Detect House, Tree, and Person elements
- Classify object sizes (small, medium, large)
- Infer psychological attributes:
  - Emotional state
  - Family relationships
  - Social circle
- Maintain user profiles and interaction history
- Deliver human-readable psychological interpretations

--------------------------------------------------
DATA COLLECTION
--------------------------------------------------

Due to the absence of publicly available datasets for HTP analysis,
all data used in this project was collected and curated manually.

HUMAN-DRAWN DATA
----------------
- Multiple in-person data collection drives
- Drawings collected from students across departments
- All drawings followed the House-Tree-Person format
- Sketches were photographed, cleaned, filtered, and digitized

Total human-drawn images retained: approximately 105

AI-GENERATED DATA
-----------------
To increase dataset diversity and robustness, AI-generated sketches
were incorporated.

Prompt used:
"Draw on a blank A4 sheet using a pencil. The drawing should look like it
was made by a 5-6 year old child."

Characteristics ensured:
- Pencil-style texture
- Child-like proportions
- Consistent HTP structure

AI-generated images: approximately 95

FINAL DATASET OVERVIEW
----------------------
Human-drawn images: ~105
AI-generated images: ~95
Total base dataset: 200 images
After augmentation: 542 images

--------------------------------------------------
DATA ANNOTATION AND AUGMENTATION
--------------------------------------------------
Annotation tool: Roboflow

Annotations:
- Bounding boxes for House, Tree, and Person

Augmentation techniques:
- Rotation
- Brightness and exposure variation
- Blur and noise
- Shear transformations

These steps significantly improved YOLO model generalization.

--------------------------------------------------
OBJECT DETECTION USING YOLOv8
--------------------------------------------------
- Model: YOLOv8
- Detection accuracy: approximately 98-99 percent
- Responsibilities:
  - Object detection
  - Area-based size classification

--------------------------------------------------
YOLO ANNOTATED OUTPUT
--------------------------------------------------

Example YOLOv8 detection result from the project:

![YOLO Annotated Output](yolo-result.jpg)

Detected elements:
- Tree   (confidence: 0.77)
- House  (confidence: 0.91)
- Person (confidence: 0.85)

Bounding box areas are later used to compute size ratios.

--------------------------------------------------
AREA-BASED SIZE CLASSIFICATION
--------------------------------------------------
Instead of comparing objects relative to each other, the system:
- Computes bounding box area divided by total image area
- Assigns size labels:
  - Small
  - Medium
  - Large

This approach enables consistent and human-aligned interpretation.

--------------------------------------------------
SYSTEM ARCHITECTURE
--------------------------------------------------

APPROACH 1
YOLO + RAG (PDF + CSV + DATABASE) + CHATBOT
--------------------------------------------------

User interface example:

![Approach 1 UI](YOLO + RAG (PDF + CSV + Database) + Chatbot.png)

Pipeline:
1. YOLO detects objects
2. RAG retrieves information from:
   - HTP psychology reference PDF
   - CSV feature dataset
   - User database
3. Chatbot generates interpretation

Strengths:
- Stable and reliable
- Maintains user identity and history
- Suitable as a baseline solution

Limitations:
- Lacks deep visual understanding
- Fully dependent on YOLO detections

--------------------------------------------------

APPROACH 2 (FINAL AND PREFERRED)
YOLO + RAG (DATABASE) + VISION MODEL + PROMPT ENGINEERING
--------------------------------------------------

User interface example:

![Approach 2 UI](YOLO + RAG (Database) + Vision Model + Prompt Engineering.png)

Pipeline:
1. YOLO performs object detection and size classification
2. Vision model analyzes the image holistically
3. Prompt engineering aligns outputs with HTP psychology
4. Database preserves user memory and interaction history

Advantages:
- Vision model compensates for YOLO detection limitations
- Context-aware and semantically rich interpretations
- Closest approximation to human psychological analysis

This is the final and production-grade system.

--------------------------------------------------
EXPERIMENTED ML MODELS AND LEARNINGS
--------------------------------------------------
- Random Forest: overfitting caused by class imbalance
- XGBoost: high accuracy but poor generalization

Key insight:
High accuracy alone does not imply meaningful reasoning.

--------------------------------------------------
TECHNOLOGY STACK
--------------------------------------------------
- Python
- YOLOv8
- OpenAI Vision Model 4.1
- Roboflow
- Flask
- RAG (Retrieval-Augmented Generation)
- Prompt Engineering
- Google Colab
- CSV handling and database integration

--------------------------------------------------
PROJECT STATUS
--------------------------------------------------
Completed.
Open to future research and extensions.

--------------------------------------------------
ACKNOWLEDGEMENTS
--------------------------------------------------
Project Supervisor:
Abdullah Sajid

Core Contributor:
Zain (Sam)

--------------------------------------------------
This repository represents a complete AI system rather than a single model.
--------------------------------------------------
