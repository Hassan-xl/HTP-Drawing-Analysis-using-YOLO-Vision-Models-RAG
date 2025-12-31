HTP DRAWING ANALYZER
Ancient Wisdom Meets Modern AI

An end-to-end AI-powered psychological drawing analysis system that interprets
House-Tree-Person (HTP) sketches using YOLOv8, Vision Models, RAG, and
Prompt Engineering.

This project evolves through two intelligent AI pipelines, moving from a
detection-based system to a hybrid vision-intelligence architecture capable of
deep psychological reasoning.

--------------------------------------------------
WHAT THIS PROJECT DOES
--------------------------------------------------
- Accepts hand-drawn HTP sketches
- Detects House, Tree, and Person
- Classifies object sizes (small / medium / large)
- Interprets:
  * Emotional State
  * Family Relationships
  * Social Circle
- Stores user profiles and chat history
- Responds like an HTP psychologist

--------------------------------------------------
DATA COLLECTION (BACKBONE OF THE PROJECT)
--------------------------------------------------

This project is built on a custom-curated dataset collected manually due to the
absence of publicly available HTP datasets.

HUMAN-DRAWN DATA
----------------
- Multiple on-campus data collection drives
- Drawings collected from students across departments
- All sketches followed the HTP (House-Tree-Person) format
- Images were:
  * Photographed manually
  * Cleaned and filtered
  * Digitized carefully to preserve drawing details

Final human-drawn images used: approximately 105

AI-GENERATED DATA
-----------------
To overcome dataset limitations, AI-generated sketches were added.

Prompt used:
"Draw on a blank A4 sheet using a pencil. The drawing should look like it was
made by a 5-6 year old child."

Ensured:
- Pencil-like texture
- Child-like proportions
- Consistent HTP structure

AI-generated images: approximately 95

FINAL DATASET SUMMARY
---------------------
Human-drawn images: ~105
AI-generated images: ~95
Total base dataset: 200
After augmentation: 542 images

--------------------------------------------------
ANNOTATION AND AUGMENTATION
--------------------------------------------------
Tool used: Roboflow

Bounding boxes created for:
- House
- Tree
- Person

Augmentations applied:
- Rotation
- Brightness and exposure adjustment
- Blur and noise
- Shear transformations

This significantly improved YOLO generalization.

--------------------------------------------------
OBJECT DETECTION (YOLOv8)
--------------------------------------------------
- Trained using YOLOv8
- Achieved approximately 98-99 percent detection accuracy
- Used exclusively for:
  * Object detection
  * Size classification

--------------------------------------------------
YOLO ANNOTATED OUTPUT (EXAMPLE)
--------------------------------------------------
Annotated image uploaded in repository:

yolo-result.jpg

Detected elements:
- Tree   (confidence: 0.77)
- House  (confidence: 0.91)
- Person (confidence: 0.85)

These detections are later used for area-ratio-based size classification.

--------------------------------------------------
INTELLIGENT SIZE CLASSIFICATION
--------------------------------------------------
Instead of comparing objects with each other, the system:
- Calculates bounding box area divided by full image area
- Assigns size labels:
  * Small
  * Medium
  * Large

This approach mimics human perception while remaining mathematically consistent.

--------------------------------------------------
SYSTEM ARCHITECTURE - TWO APPROACHES
--------------------------------------------------

APPROACH 1
YOLO + RAG (PDF + CSV + DATABASE) + CHATBOT
--------------------------------------------------

UI Screenshot (uploaded in repository):
YOLO + RAG (PDF + CSV + Database) + Chatbot.png

Pipeline:
1. YOLO detects objects
2. RAG retrieves:
   - HTP psychology PDF
   - CSV dataset
   - User database
3. Chatbot generates interpretation

Pros:
- Stable
- Memory-aware (user profiles and history)
- Good baseline performance (approximately 85 percent)

Cons:
- No deep visual understanding
- Fully dependent on YOLO detections

Reliable, but context-limited.

--------------------------------------------------

APPROACH 2 (FINAL AND PREFERRED)
YOLO + RAG (DATABASE) + VISION MODEL + PROMPT ENGINEERING
--------------------------------------------------

UI Screenshot (uploaded in repository):
YOLO + RAG (Database) + Vision Model + Prompt Engineering.png

Pipeline:
1. YOLO performs object detection and size classification
2. Vision model performs full image understanding
3. Prompt engineering enforces HTP psychologist behavior
4. Database stores user memory and interaction history

Why this works better:
- Vision model compensates for YOLO misses
- Rich psychological interpretation
- Context-aware reasoning
- Closest to human-level analysis

This is the final production-grade system.

--------------------------------------------------
FAILED MODELS AND KEY LEARNINGS
--------------------------------------------------
- Random Forest: overfitting due to class imbalance
- XGBoost: high accuracy but poor generalization

Key lesson:
Accuracy is meaningless without reasoning.

--------------------------------------------------
TECH STACK
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
Completed
Open for future research and improvements

--------------------------------------------------
ACKNOWLEDGEMENTS
--------------------------------------------------
Project Supervisor:
Abdullah Sajid

Core Contributor:
Zain (Sam)

--------------------------------------------------
If this project impressed you, consider giving it a star.
This is not just a model, it is a complete AI system.
--------------------------------------------------
