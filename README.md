# HTP Drawing Analyzer
## Ancient Wisdom Meets Modern AI

HTP Drawing Analyzer is an AI-powered system designed to analyze
House-Tree-Person (HTP) drawings and generate structured psychological
interpretations. The system combines object detection, vision-based
reasoning, retrieval-augmented generation (RAG), and prompt engineering
to simulate the analytical behavior of an HTP psychologist.

This repository presents two complete AI pipelines, showcasing the
evolution from a detection-centric approach to a hybrid vision-based
intelligent system.

--------------------------------------------------
PROJECT HIGHLIGHTS
--------------------------------------------------
- End-to-end AI system (not just a model)
- Custom-collected HTP dataset
- YOLOv8-based object detection
- Vision-model-driven psychological reasoning
- Memory-aware user interaction
- Two distinct architectural approaches

--------------------------------------------------
DATA COLLECTION
--------------------------------------------------

Due to the lack of publicly available datasets for HTP analysis, the
entire dataset used in this project was collected and curated manually.

HUMAN-DRAWN DATA
----------------
- Multiple on-campus data collection drives
- Drawings collected from students across departments
- All sketches followed the House-Tree-Person format
- Images were photographed, cleaned, filtered, and digitized manually

Total human-drawn images retained: approximately 105

AI-GENERATED DATA
-----------------
To improve robustness and diversity, AI-generated sketches were added.

Prompt used:
"Draw on a blank A4 sheet using a pencil. The drawing should look like it
was made by a 5-6 year old child."

AI-generated images: approximately 95

FINAL DATASET SUMMARY
---------------------
Human-drawn images: ~105
AI-generated images: ~95
Total base dataset: 200 images
After augmentation: 542 images

--------------------------------------------------
ANNOTATION AND AUGMENTATION
--------------------------------------------------
- Tool used: Roboflow
- Bounding boxes created for:
  - House
  - Tree
  - Person
- Augmentations applied:
  - Rotation
  - Brightness and exposure
  - Blur and noise
  - Shear transformations

--------------------------------------------------
YOLOv8 OBJECT DETECTION
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

![YOLO Annotated Output](images/yolo-result.jpg)

Detected elements:
- Tree   (confidence: 0.77)
- House  (confidence: 0.91)
- Person (confidence: 0.85)

--------------------------------------------------
INTELLIGENT SIZE CLASSIFICATION
--------------------------------------------------
Instead of comparing objects with each other, the system:
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

User Interface Preview:

![Approach 1 UI](images/approach1_ui.png)

Pipeline:
1. YOLO detects objects in the drawing
2. RAG retrieves information from:
   - HTP psychology reference PDF
   - CSV feature dataset
   - User database
3. Chatbot generates the final interpretation

Strengths:
- Stable and reliable
- Maintains user identity and chat history
- Suitable as a baseline analytical system

Limitations:
- No deep visual understanding
- Fully dependent on YOLO detections

--------------------------------------------------

APPROACH 2 (FINAL AND PREFERRED)
YOLO + RAG (DATABASE) + VISION MODEL + PROMPT ENGINEERING
--------------------------------------------------

User Interface Preview:

![Approach 2 UI](images/approach2_ui.png)

Pipeline:
1. YOLO performs object detection and size classification
2. Vision model analyzes the drawing holistically
3. Prompt engineering enforces HTP psychologist reasoning
4. Database preserves user memory and interaction history

Why this approach is superior:
- Vision model compensates for YOLO detection limitations
- Rich, context-aware psychological interpretation
- Closest approximation to human-level analysis

This is the final production-grade system.

--------------------------------------------------
EXPERIMENTED MODELS AND KEY LEARNINGS
--------------------------------------------------
- Random Forest: failed due to class imbalance and overfitting
- XGBoost: high accuracy but poor generalization

Key insight:
High accuracy does not imply meaningful intelligence.

--------------------------------------------------
TECHNOLOGY STACK
--------------------------------------------------
- Python
- YOLOv8
- OpenAI Vision Model 4.1
- Roboflow
- Flask
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Google Colab
- CSV handling and database integration

--------------------------------------------------
PROJECT STATUS
--------------------------------------------------
Completed.
Open for future research, scaling, and improvements.

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
