from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import base64, json, os, secrets, csv
from datetime import datetime
from openai import OpenAI
import re
import pdfplumber

client = OpenAI(api_key="")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///htp_analyzer.db'
app.config['SECRET_KEY'] = secrets.token_hex(16)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')
    memories = db.relationship('UserMemory', backref='user', lazy=True, cascade='all, delete-orphan')

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    yolo_data = db.Column(db.JSON, nullable=False)
    gpt_analysis = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('ChatMessage', backref='analysis', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserMemory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    memory_type = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

with app.app_context():
    db.create_all()

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

HTP_DATASET = []
PDF_INTERPRETATIONS = {}

def load_rag_knowledge():
    global HTP_DATASET, PDF_INTERPRETATIONS
    
    try:
        csv_path = 'data/htp_dataset.csv'
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            HTP_DATASET = list(reader)
            print(f"✓ Loaded {len(HTP_DATASET)} HTP profiles from {csv_path}")
    except FileNotFoundError:
        print(f"✗ CSV not found at {csv_path}")
        HTP_DATASET = []
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
    
    try:
        pdf_path = 'data/htp_interpretations.pdf'
        with pdfplumber.open(pdf_path) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() + "\n"
            PDF_INTERPRETATIONS['raw_text'] = pdf_text[:1500]
            print(f"✓ Loaded PDF interpretations from {pdf_path}")
    except FileNotFoundError:
        print(f"✗ PDF not found at {pdf_path}")
    except Exception as e:
        print(f"✗ Error loading PDF: {e}")

load_rag_knowledge()

def find_similar_htp_profiles(yolo_data):
    if not HTP_DATASET:
        return []
    similar = []
    for row in HTP_DATASET[:20]:
        try:
            if row.get('person_size') and row.get('emotional_state'):
                similar.append(row)
        except:
            continue
    return similar[:5]

def extract_and_save_memory(user_id, message):
    """Extract and save memories STRICTLY PER USER_ID with improved nickname detection"""
    patterns = {
        'nickname': [
            r"(?:call me|my name is|i'm|i am|just call me)\s+([A-Za-z]+)",
            r"(?:you can call me|name me|names?\s+(?:is|are))\s+([A-Za-z]+)",
            r"(?:i go by|people call me)\s+([A-Za-z]+)",
            r"(?:nickname|nick)\s+(?:is|=)\s+([A-Za-z]+)"
        ],
        'favorite_food': r"(?:favorite|fav|love|enjoy)\s+(?:food|eating|meal)[\s:]+([A-Za-z\s]+?)(?:\.|,|$)",
        'hobby': r"(?:hobby|hobbies|love|enjoy|like|passion)[\s:]+([A-Za-z\s]+?)(?:\.|,|$)",
        'trait': r"(?:i'm|i am|i feel|see myself as)[\s:]+(?:very\s+)?([A-Za-z\s]+?)(?:\.|,|$)"
    }
    
    for memory_type, pattern_list in patterns.items():
        patterns_to_check = pattern_list if isinstance(pattern_list, list) else [pattern_list]
        
        for pattern in patterns_to_check:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if len(content) > 2:
                    existing = UserMemory.query.filter_by(
                        user_id=user_id, 
                        memory_type=memory_type
                    ).first()
                    
                    if existing:
                        existing.content = content
                        existing.updated_at = datetime.utcnow()
                    else:
                        memory = UserMemory(
                            user_id=user_id,
                            memory_type=memory_type,
                            content=content
                        )
                        db.session.add(memory)
                    db.session.commit()
                    print(f"[v0] Saved {memory_type}: {content} for user {user_id}")
                    break

def build_rag_prompt(final_output, user_id, username):
    memories = UserMemory.query.filter_by(user_id=user_id).all()
    memory_context = ""
    if memories:
        memory_context = "\n\nUSER MEMORIES (Remember these about them):\n"
        for mem in memories:
            memory_context += f"- {mem.memory_type}: {mem.content}\n"

    similar_profiles = find_similar_htp_profiles(final_output)
    similar_context = ""
    if similar_profiles:
        similar_context = "\n\nSIMILAR PROFILES FROM DATASET:\n"
        for profile in similar_profiles[:3]:
            similar_context += f"- Person Size: {profile.get('person_size')}, Emotional State: {profile.get('emotional_state')}\n"
    
    pdf_context = "\n\nHTP INTERPRETATION GUIDE:\n"
    if PDF_INTERPRETATIONS.get('raw_text'):
        pdf_context += PDF_INTERPRETATIONS['raw_text']
    else:
        pdf_context += "House=family/privacy, Tree=confidence/growth, Person=self-image\n"
    
    return f"""
You are the Pharao, ancient Egyptian wisdom keeper analyzing {username}'s HTP drawing.
Keep responses SHORT and FRIENDLY - max 2-3 sentences, like texting a friend.
Be warm, use their name, feel like a real friend.

IMPORTANT: These are {username}'s memories. Reference them naturally.{memory_context}

YOLO MEASUREMENTS:
{json.dumps(final_output, indent=2)}

{pdf_context}{similar_context}

Remember: SHORT, simple, friendly - NOT textbook paragraphs.
Speak like texting them, use their memories, be genuine.
"""

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already exists"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already exists"}), 400

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        return jsonify({"success": True, "user_id": user.id})

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({"error": "Invalid credentials"}), 401

        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({"success": True, "redirect": url_for("index")})

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_id = session['user_id']
        
        file = request.files["image"]
        image_path = f"static/uploads/{user_id}_{datetime.utcnow().timestamp()}.jpg"
        os.makedirs("static/uploads", exist_ok=True)
        file.save(image_path)

        results = model(image_path)[0]
        img_h, img_w = results.orig_img.shape[:2]

        annotated_path = f"static/annotated/{user_id}_{datetime.utcnow().timestamp()}.jpg"
        os.makedirs("static/annotated", exist_ok=True)
        results.save(filename=annotated_path)

        raw_output = {}
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

            raw_output[label] = {
                "ratio": round(ratio, 6),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            }

        def classify_size(r):
            if r > 0.09: return "LARGE"
            elif r > 0.06: return "MEDIUM"
            else: return "SMALL"

        final_output = {
            label: {
                "size": classify_size(det["ratio"]),
                "ratio": det["ratio"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            for label, det in raw_output.items()
        }

        prompt = build_rag_prompt(final_output, user_id, session['username'])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze this HTP drawing and give me a brief insight about myself."}
            ],
            max_tokens=150
        )

        analysis = response.choices[0].message.content

        analysis_record = Analysis(
            user_id=user_id,
            yolo_data=final_output,
            gpt_analysis=analysis,
            image_path=annotated_path
        )
        db.session.add(analysis_record)
        db.session.commit()

        htp_chart_data = {
            "labels": list(final_output.keys()),
            "sizes": [final_output[label]["size"] for label in final_output.keys()],
            "ratios": [round(final_output[label]["ratio"] * 100, 2) for label in final_output.keys()]
        }

        return render_template(
            "index.html",
            output=analysis,
            annotated_image=annotated_path,
            yolo_json=json.dumps(final_output, indent=2),
            analysis_id=analysis_record.id,
            username=session['username'],
            htp_chart_data=json.dumps(htp_chart_data)
        )

    user = User.query.get(session['user_id'])
    return render_template("index.html", username=session['username'], user=user)

@app.route("/api/htp-chart/<int:analysis_id>", methods=["GET"])
def get_htp_chart(analysis_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    analysis = Analysis.query.get(analysis_id)
    if not analysis or analysis.user_id != session['user_id']:
        return jsonify({"error": "Analysis not found"}), 404
    
    yolo_data = analysis.yolo_data
    chart_data = {
        "labels": list(yolo_data.keys()),
        "sizes": [yolo_data[label]["size"] for label in yolo_data.keys()],
        "ratios": [round(yolo_data[label]["ratio"] * 100, 2) for label in yolo_data.keys()]
    }
    
    return jsonify(chart_data)

@app.route("/api/chat", methods=["POST"])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_message = data.get("message")
    analysis_id = data.get("analysis_id")

    user = User.query.get(session['user_id'])
    username = user.username if user else "seeker"
    user_id = session['user_id']

    analysis = Analysis.query.get(analysis_id)
    if not analysis or analysis.user_id != user_id:
        return jsonify({"error": "Analysis not found"}), 404

    extract_and_save_memory(user_id, user_message)

    memories = UserMemory.query.filter_by(user_id=user_id).all()
    memory_context = ""
    if memories:
        memory_context = "\nUSER MEMORIES: "
        for mem in memories:
            memory_context += f"({mem.memory_type}: {mem.content}) "

    similar_profiles = find_similar_htp_profiles(analysis.yolo_data)
    similar_text = ""
    if similar_profiles:
        similar_text = "\nSimilar profiles: " + ", ".join([p.get('emotional_state', 'unknown') for p in similar_profiles[:2]])

    pharao_context = f"""
You are the Pharao - warm, friendly HTP guide speaking to {username}.
KEEP RESPONSES SHORT - max 1-2 sentences, like texting a friend!
Reference their memories naturally if relevant.{memory_context}{similar_text}

THEIR DRAWING ANALYSIS:
{analysis.gpt_analysis}

Remember: SHORT and FRIENDLY, not formal.
Respond like a real friend who cares about them.

{username} says: {user_message}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": pharao_context},
            {"role": "user", "content": user_message}
        ],
        max_tokens=100
    )

    pharao_reply = response.choices[0].message.content

    user_msg = ChatMessage(
        user_id=user_id,
        analysis_id=analysis_id,
        role='user',
        message=user_message
    )
    pharao_msg = ChatMessage(
        user_id=user_id,
        analysis_id=analysis_id,
        role='pharao',
        message=pharao_reply
    )
    db.session.add(user_msg)
    db.session.add(pharao_msg)
    db.session.commit()

    return jsonify({"reply": pharao_reply})

@app.route("/api/user-memories", methods=["GET"])
def get_user_memories():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    memories = UserMemory.query.filter_by(user_id=session['user_id']).all()
    return jsonify([{
        "type": m.memory_type,
        "content": m.content,
        "updated": m.updated_at.isoformat()
    } for m in memories])

if __name__ == "__main__":
    app.run(debug=True)
