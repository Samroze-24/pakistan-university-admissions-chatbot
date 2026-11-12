"""Flask web app: PakUni Guide — Pakistan University Admissions Chatbot.

Routes:
    GET  /                      -- the chat page
    GET  /api/sample-questions  -- example questions to try
    GET  /api/stats             -- intent/KB/model numbers
    POST /chat                  -- JSON in, bot reply out

How it works (three honest pieces, kept separate):
  * Intent classification (ML): a TF-IDF + LogisticRegression model trained
    on sample utterances decides what the user is asking about
    (13 intents). Short, unambiguous messages (greetings, goodbyes, and any
    message containing "merit") are routed by a small deterministic keyword
    router first — a classifier is overkill for "hello", and it is also
    unreliable on such short inputs with so little training data.
    Sample utterances are templates, not real user messages,
    so accuracy on real phrasing will be lower — the README says so.
  * Answer retrieval (TF-IDF cosine): the reply comes from a curated
    knowledge base of 30 Q&A pairs (data/kb.json). The bot never invents
    an answer: if nothing matches well enough, it says so and suggests
    questions it can answer.
  * Merit calculator (multi-turn slot filling): for merit questions the bot
    asks for Intermediate % and entry-test %, then computes merit with a
    labelled sample 70/30 formula. The numbers are illustrative, not
    official cut-offs.
  * All fees, cut-offs and dates in the KB are labelled sample/illustrative.
    Always confirm exact values on the university's official site.
"""

import json
import re
import uuid
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "kb.json"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

app = Flask(__name__)

# Sample merit weights — labelled as a sample formula everywhere it is used.
MERIT_INTER_WEIGHT = 0.7
MERIT_TEST_WEIGHT = 0.3

GREETING_REPLY = (
    "Assalam o alaikum! I'm PakUni Guide — I can help with university admissions "
    "in Pakistan: deadlines, merit calculation, entry tests, fees, scholarships, "
    "programs, documents, hostels and eligibility. Try the chips below, or ask me "
    "something like 'mera merit kitna banega?'"
)
SMALLTALK_REPLIES = {
    "name": "I'm PakUni Guide, an admissions chatbot built with Flask and "
            "scikit-learn. I answer from a small sample knowledge base — I'm not "
            "a real person and not an official university helpline.",
    "how_are_you": "I'm running fine, thank you! How can I help with admissions?",
    "who_made": "I was built as a sample AI project with Flask, scikit-learn and "
                "a curated Q&A knowledge base. My answers are general information, "
                "not official university notices.",
    "what_can_you_do": "I can answer questions about admissions deadlines, merit, "
                       "entry tests, fees, scholarships, programs, documents, "
                       "hostels and eligibility — and I can calculate your merit "
                       "if you give me your Intermediate and entry-test percentages.",
    "thanks": "You're welcome! Anything else about admissions?",
}
GOODBYE_REPLY = (
    "Allah Hafiz! Good luck with your admission. If you need help later, "
    "I'm here — and do confirm final dates and fees on the university's official site."
)
FALLBACK_REPLY = (
    "I don't have a confident answer for that yet — my knowledge base is small "
    "and sample-only. Try one of the suggested questions below, or ask about "
    "deadlines, merit, entry tests, fees, scholarships, programs, documents, "
    "hostels or eligibility."
)
SUGGESTIONS = [
    "mera merit kitna banega?", "fee structure for BSCS",
    "scholarships available?", "entry test pattern kya hai",
    "documents kon se chahiye", "hostel available hai?",
]

_model = None
_metrics = None
_kb = None
_kb_vectorizer = None
_kb_matrix = None
_sessions = {}


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("model.pkl not found. Run `python train_model.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None and METRICS_PATH.exists():
        _metrics = json.loads(METRICS_PATH.read_text())
    return _metrics or {}


def get_kb():
    global _kb
    if _kb is None:
        _kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    return _kb


def get_kb_index():
    global _kb_vectorizer, _kb_matrix
    if _kb_vectorizer is None:
        kb = get_kb()
        texts = [f"{e['question']} {' '.join(e['keywords'])}" for e in kb]
        _kb_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        _kb_matrix = _kb_vectorizer.fit_transform(texts)
    return _kb_vectorizer, _kb_matrix


def route_keywords(message):
    """Deterministic routing for greetings, goodbyes and merit questions.

    Returns an intent, or None to fall through to the ML classifier.
    Keyword routing is used where it is unambiguous: a greeting is a
    greeting, and any question about merit belongs to the merit flow
    (which then separates "merit list" questions from calculations).
    """
    low = message.lower().strip()
    words = set(re.findall(r"[a-z]+", low))
    if words & {"hello", "hey", "hi", "salam", "salaam", "aoa"} or low in {
            "good morning", "good afternoon", "good evening", "morning"}:
        return "greeting"
    if words & {"bye", "alvida"} or "allah hafiz" in low or "khuda hafiz" in low \
            or "goodbye" in low or "good bye" in low or "good night" in low:
        return "goodbye"
    if "merit" in low:
        return "merit_calculation"
    if len(numbers_in(message)) >= 2 and ("inter" in low or "test" in low):
        # "inter 90 test 80" — two percentages plus inter/test means a merit
        # calculation, even without the word "merit".
        return "merit_calculation"
    return None


def classify(message):
    """Returns (intent, confidence) from the ML classifier."""
    model = get_model()
    proba = model.predict_proba([message])[0]
    idx = int(proba.argmax())
    return model.classes_[idx], float(proba[idx])


def retrieve(message, intent=None):
    """Best KB entry by TF-IDF cosine. Returns (entry, score)."""
    kb = get_kb()
    vectorizer, matrix = get_kb_index()
    vec = vectorizer.transform([message])
    sims = cosine_similarity(vec, matrix)[0]
    if intent:
        # Prefer an entry of the classified intent when it is close.
        best_in_intent = max(
            ((sims[i], i) for i, e in enumerate(kb) if e["intent"] == intent),
            default=(0.0, -1),
        )
        if best_in_intent[1] >= 0 and best_in_intent[0] >= 0.15:
            return kb[best_in_intent[1]], float(best_in_intent[0])
    best = int(sims.argmax())
    return kb[best], float(sims[best])


def numbers_in(text):
    """Percent-like numbers (0-100) found in the message, in order."""
    return [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)", text)
            if 0 <= float(n) <= 100]


def merit_reply(inter_pct, test_pct):
    merit = MERIT_INTER_WEIGHT * inter_pct + MERIT_TEST_WEIGHT * test_pct
    if merit >= 85:
        verdict = ("Very strong — in the sample guide this range competes for top "
                   "public universities.")
    elif merit >= 70:
        verdict = ("Competitive — in the sample guide this range fits mid-tier public "
                   "universities and good private ones.")
    elif merit >= 50:
        verdict = ("Workable — in the sample guide this range fits many private "
                   "universities and some public programs.")
    else:
        verdict = ("Below the 50% minimum most programs ask for in the sample guide — "
                   "check the eligibility rules of your target university.")
    return (
        f"Your merit (sample 70/30 formula: 0.7 x Inter + 0.3 x Test) is "
        f"{merit:.1f}% — from Inter {inter_pct:g}% and entry test {test_pct:g}%. "
        f"{verdict} Remember: real weights and cut-offs vary by university, so "
        f"confirm the official formula before deciding."
    )


def get_session(session_id):
    if not session_id or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        _sessions[session_id] = {"awaiting": None, "inter_pct": None}
    return session_id, _sessions[session_id]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample-questions")
def sample_questions():
    return jsonify({"success": True, "questions": SUGGESTIONS})


@app.route("/api/stats")
def stats():
    kb = get_kb()
    return jsonify({
        "success": True,
        "intents": sorted({e["intent"] for e in kb} | {"greeting", "smalltalk", "goodbye"}),
        "kb_entries": len(kb),
        "model_metrics": get_metrics(),
        "merit_formula": "Sample: 0.7 x Inter % + 0.3 x Entry-test %",
    })


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"success": False, "error": "Message is required."}), 400
    if len(message) > 500:
        return jsonify({"success": False, "error": "Message is too long (max 500 characters)."}), 400
    session_id, session = get_session(str(payload.get("session_id", "") or ""))

    # Multi-turn merit flow: we are waiting for a percentage.
    if session["awaiting"] == "inter_pct":
        nums = numbers_in(message)
        if not nums:
            return jsonify({"success": True, "session_id": session_id,
                            "reply": "Please send your Intermediate percentage as a number, e.g. 85.",
                            "intent": "merit_calculation", "confidence": None,
                            "suggestions": SUGGESTIONS})
        session["inter_pct"] = nums[0]
        session["awaiting"] = "test_pct"
        return jsonify({"success": True, "session_id": session_id,
                        "reply": f"Got it — Inter {nums[0]:g}%. Now send your entry-test percentage (0-100).",
                        "intent": "merit_calculation", "confidence": None,
                        "suggestions": SUGGESTIONS})
    if session["awaiting"] == "test_pct":
        nums = numbers_in(message)
        if not nums:
            return jsonify({"success": True, "session_id": session_id,
                            "reply": "Please send your entry-test percentage as a number, e.g. 70.",
                            "intent": "merit_calculation", "confidence": None,
                            "suggestions": SUGGESTIONS})
        reply = merit_reply(session["inter_pct"], nums[0])
        session["awaiting"] = None
        session["inter_pct"] = None
        return jsonify({"success": True, "session_id": session_id,
                        "reply": reply, "intent": "merit_calculation",
                        "confidence": None, "suggestions": SUGGESTIONS})

    intent = route_keywords(message)
    if intent is not None:
        confidence = None  # rule-routed, not an ML confidence
    else:
        intent, confidence = classify(message)
    low = message.lower()
    # A low-confidence label is not trusted: the reply then comes from KB
    # retrieval, or the honest fallback — never from a canned branch.
    trusted = confidence is None or confidence >= 0.35

    if trusted and intent == "greeting":
        reply = GREETING_REPLY
    elif trusted and intent == "goodbye":
        reply = GOODBYE_REPLY
    elif trusted and intent == "smalltalk":
        if "name" in low or "naam" in low:
            reply = SMALLTALK_REPLIES["name"]
        elif "how are you" in low or "kaise ho" in low:
            reply = SMALLTALK_REPLIES["how_are_you"]
        elif "who" in low or "kis ne" in low or "banaya" in low:
            reply = SMALLTALK_REPLIES["who_made"]
        elif "thank" in low or "shukriya" in low:
            reply = SMALLTALK_REPLIES["thanks"]
        else:
            reply = SMALLTALK_REPLIES["what_can_you_do"]
    elif trusted and intent == "merit_calculation":
        if "list" in low:
            # "merit list kab aati hai" is a question, not a calculation.
            entry, _score = retrieve(message, intent=intent)
            reply = entry["answer"]
        else:
            nums = numbers_in(message)
            if len(nums) >= 2:
                reply = merit_reply(nums[0], nums[1])
            else:
                session["awaiting"] = "inter_pct"
                if len(nums) == 1:
                    session["inter_pct"] = nums[0]
                    session["awaiting"] = "test_pct"
                    reply = (f"Got it — Inter {nums[0]:g}%. Now send your entry-test "
                             f"percentage (0-100).")
                else:
                    reply = ("I can calculate your merit. What is your Intermediate "
                             "percentage? Send it as a number, e.g. 85.")
    else:
        # Untrusted label: retrieve without preferring that intent.
        entry, score = retrieve(message, intent=intent if trusted else None)
        if (confidence or 0) >= 0.6 or score >= 0.25:
            reply = entry["answer"]
        else:
            reply = FALLBACK_REPLY

    return jsonify({"success": True, "session_id": session_id, "reply": reply,
                    "intent": intent,
                    "confidence": round(confidence, 3) if confidence is not None else None,
                    "suggestions": SUGGESTIONS})


if __name__ == "__main__":
    app.run(debug=True)
