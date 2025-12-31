from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import base64, json, os, secrets, re
from datetime import datetime
from openai import OpenAI

# -------------------------
# 🔐 API KEY
# -------------------------
client = OpenAI(api_key="")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///htp_analyzer.db'
app.config['SECRET_KEY'] = secrets.token_hex(16)
db = SQLAlchemy(app)

# -------------------------
# 📊 Database Models
# -------------------------
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


# -------------------------
# 🧱 Memory Extraction Logic
# -------------------------
def extract_and_save_memory(user_id, message):
    patterns = {
        'nickname': [
            r"(?:call me|my name is|i'm|i am|just call me)\s+([A-Za-z]+)",
            r"(?:you can call me|people call me|name me)\s+([A-Za-z]+)"
        ],
        'favorite_food': r"(?:favorite|fav|love|enjoy)\s+(?:food|meal)[\s:]+([A-Za-z\s]+?)(?:\.|,|$)",
        'hobby': r"(?:hobby|hobbies|love|enjoy|like)[\s:]+([A-Za-z\s]+?)(?:\.|,|$)",
        'trait': r"(?:i'm|i am|i feel|i tend to be)[\s:]+([A-Za-z\s]+?)(?:\.|,|$)",
    }

    for mem_type, pattern_list in patterns.items():
        patterns_to_use = pattern_list if isinstance(pattern_list, list) else [pattern_list]

        for pattern in patterns_to_use:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                content = match.group(1).strip()

                existing = UserMemory.query.filter_by(
                    user_id=user_id,
                    memory_type=mem_type
                ).first()

                if existing:
                    existing.content = content
                    existing.updated_at = datetime.utcnow()
                else:
                    mem = UserMemory(
                        user_id=user_id,
                        memory_type=mem_type,
                        content=content
                    )
                    db.session.add(mem)

                db.session.commit()
                return


# -------------------------
# 🧠 Load YOLO model
# -------------------------
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)


# -------------------------
# 📦 Build HTP Prompt (Strict 3 sections + Final Summary)
# -------------------------
def build_htp_prompt(final_output):
    return f"""
You are an HTP-style interpreter.  
Use ONLY the YOLO-detected sizes (LARGE, MEDIUM, SMALL) for House, Tree, and Person.

STRICT RULES:
- Output MUST contain exactly these four sections:
  1. Family Relationships
  2. Social Circle
  3. Emotional Tone
  4. Final Summary
- Each of the first three MUST be 1–2 sentences.
- The Final Summary MUST be exactly one short sentence.
- No long paragraphs. No symbolic over-explaining. No diagnosing.

SIZE MEANINGS:
- LARGE → strong presence, importance, emotional weight  
- MEDIUM → balanced presence, moderate meaning  
- SMALL → subtle, quiet, or vulnerable meaning  

FORMAT (STRICT):

**Family Relationships:**  
(1–2 sentences using HOUSE size)

**Social Circle:**  
(1–2 sentences using TREE size)

**Emotional Tone:**  
(1–2 sentences using PERSON size)

**Final Summary:**  
(1 short sentence connecting all three)

YOLO DATA:
{json.dumps(final_output, indent=2)}
"""


# -------------------------
# Auth System
# -------------------------
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


# -------------------------
# HTP Analysis (Main Page)
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        user_id = session["user_id"]

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
                "bbox": [x1, y1, x2, y2],
            }

        def classify_size(r):
            if r > 0.09:
                return "LARGE"
            if r > 0.06:
                return "MEDIUM"
            return "SMALL"

        final_output = {
            label: {
                "size": classify_size(det["ratio"]),
                "ratio": det["ratio"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
            for label, det in raw_output.items()
        }

        # Encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = build_htp_prompt(final_output)

        response = client.responses.create(
            model="gpt-4o-2024-11-20",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                    ],
                }
            ],
        )

        analysis_text = response.output_text

        record = Analysis(
            user_id=user_id,
            yolo_data=final_output,
            gpt_analysis=analysis_text,
            image_path=annotated_path,
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            "index.html",
            output=analysis_text,
            annotated_image=annotated_path,
            yolo_json=json.dumps(final_output, indent=2),
            analysis_id=record.id,
            username=session["username"],
        )

    return render_template("index.html", username=session["username"])


# -------------------------
# 💬 CHAT SYSTEM (Companion with Full RAG Memory)
# -------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    user_message = data["message"]
    analysis_id = data["analysis_id"]
    user_id = session["user_id"]

    user = User.query.get(user_id)
    username = user.username

    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404

    extract_and_save_memory(user_id, user_message)

    # FULL user chat memory (RAG)
    history_items = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.created_at.asc()).all()

    history_str = ""
    for msg in history_items:
        role = "You" if msg.role == "user" else "Companion"
        history_str += f"{role}: {msg.message}\n"

    # Load structured memories
    memories = UserMemory.query.filter_by(user_id=user_id).all()
    memory_text = "".join(f"- {m.memory_type}: {m.content}\n" for m in memories)

    # Try to get nickname from memory
    nickname = None
    for m in memories:
        if m.memory_type == "nickname":
            nickname = m.content
            break

    if nickname is None:
        nickname_display = "None saved yet"
    else:
        nickname_display = nickname

    # -------------------------
    # Persona: warm 1–2 sentence companion
    # -------------------------
    persona = f"""
You are a warm, emotionally grounded companion.
Speak softly, humanly, and with presence.
Your reply MUST be 1–2 gentle sentences.

USER REAL NAME: {username}
USER NICKNAME: {nickname_display}

RULES:
- If a nickname exists, you may use it gently.
- Never guess unknown facts.
- Never reinterpret the drawing again.
- Always stay aligned with the SACRED ANALYSIS below.
- Use memory softly but meaningfully.
- Use past chat history to maintain emotional continuity.
- No paragraphs. No mystical tone. No clinical terms.

USER MEMORIES:
{memory_text}

FULL CONVERSATION HISTORY:
{history_str}

SACRED ANALYSIS (must guide emotional tone):
{analysis.gpt_analysis}

The user says: "{user_message}"

Respond with only 1–2 warm, grounded sentences.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": user_message},
        ],
        max_tokens=80,
    )

    reply = response.choices[0].message.content

    new_user_msg = ChatMessage(
        user_id=user_id, analysis_id=analysis_id, role="user", message=user_message
    )
    new_bot_msg = ChatMessage(
        user_id=user_id, analysis_id=analysis_id, role="pharao", message=reply
    )

    db.session.add(new_user_msg)
    db.session.add(new_bot_msg)
    db.session.commit()

    return jsonify({"reply": reply})


# -------------------------
# ▶ RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
