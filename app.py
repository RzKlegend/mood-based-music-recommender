from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# PYTHON CONCEPTS: Object creation and keyword arguments.
app = Flask(__name__, template_folder=".", static_folder=".")

# PYTHON CONCEPTS: List of dictionaries (nested data structure).
MUSIC_DATABASE = [
    {"title": "Walking on Sunshine", "artist": "Katrina & The Waves", "mood": "happy", "genres": ["pop", "upbeat"]},
    {"title": "Good As Hell", "artist": "Lizzo", "mood": "happy", "genres": ["pop", "upbeat"]},
    {"title": "Don't Stop Me Now", "artist": "Queen", "mood": "happy", "genres": ["rock", "upbeat"]},
    {"title": "Here Comes the Sun", "artist": "The Beatles", "mood": "happy", "genres": ["rock", "upbeat"]},
    {"title": "Happy", "artist": "Pharrell Williams", "mood": "happy", "genres": ["pop", "upbeat"]},
    {"title": "Someone Like You", "artist": "Adele", "mood": "sad", "genres": ["ballad", "emotional"]},
    {"title": "The Night We Met", "artist": "Lord Huron", "mood": "sad", "genres": ["indie", "melancholic"]},
    {"title": "Hurt", "artist": "Johnny Cash", "mood": "sad", "genres": ["country", "emotional"]},
    {"title": "Black", "artist": "Pearl Jam", "mood": "sad", "genres": ["grunge", "melancholic"]},
    {"title": "Sad Beautiful Tragic", "artist": "Taylor Swift", "mood": "sad", "genres": ["pop", "emotional"]},
    {"title": "Weightless", "artist": "Marconi Union", "mood": "calm", "genres": ["ambient", "relaxing"]},
    {"title": "Breathe", "artist": "Pink Floyd", "mood": "calm", "genres": ["progressive", "meditative"]},
    {"title": "Clair de Lune", "artist": "Claude Debussy", "mood": "calm", "genres": ["classical", "serene"]},
    {"title": "Skinny Love", "artist": "Bon Iver", "mood": "calm", "genres": ["indie", "acoustic"]},
    {"title": "All Is Well", "artist": "Edward Sharpe", "mood": "calm", "genres": ["folk", "peaceful"]},
    {"title": "Eye of the Tiger", "artist": "Survivor", "mood": "energetic", "genres": ["rock", "intense"]},
    {"title": "High for This", "artist": "The Weeknd", "mood": "energetic", "genres": ["electronic", "intense"]},
    {"title": "Shut Up and Dance", "artist": "Walk the Moon", "mood": "energetic", "genres": ["pop", "intense"]},
    {"title": "Blinding Lights", "artist": "The Weeknd", "mood": "energetic", "genres": ["synth", "intense"]},
    {"title": "Thunderstruck", "artist": "AC/DC", "mood": "energetic", "genres": ["rock", "intense"]},
    {"title": "Thinking Out Loud", "artist": "Ed Sheeran", "mood": "romantic", "genres": ["pop", "emotional"]},
    {"title": "Perfect", "artist": "Ed Sheeran", "mood": "romantic", "genres": ["pop", "romantic"]},
    {"title": "All of Me", "artist": "John Legend", "mood": "romantic", "genres": ["pop", "emotional"]},
    {"title": "A Thousand Years", "artist": "Christina Perri", "mood": "romantic", "genres": ["pop", "romantic"]},
    {"title": "Chasing Cars", "artist": "Snow Patrol", "mood": "romantic", "genres": ["indie", "emotional"]},
]

# PYTHON CONCEPTS: Dictionary whose values are lists.
MOOD_KEYWORDS = {
    "happy": ["happy", "great", "good", "awesome", "excited", "joy", "smile", "promoted", "win", "celebrate", "proud", "fun"],
    "sad": ["sad", "down", "depressed", "heartbroken", "cry", "lonely", "lost", "hurt", "upset", "disappointed", "hopeless", "broken"],
    "calm": ["calm", "peaceful", "relaxed", "chill", "quiet", "slow", "sleepy", "exhausted", "tired", "rest", "meditative", "rain"],
    "energetic": ["energetic", "hyped", "pumped", "motivated", "fired up", "angry", "rage", "anxious", "stressed", "intense", "focus", "power"],
    "romantic": ["love", "romantic", "miss", "crush", "affection", "partner", "girlfriend", "boyfriend", "beautiful", "together", "heart", "adore"],
}

# PYTHON CONCEPTS: List of tuples for supervised ML training data.
TRAINING_TEXTS_AND_MOODS = [
    ("I am so happy and excited today", "happy"),
    ("Just got promoted at work, feeling great", "happy"),
    ("Had the best day ever", "happy"),
    ("Feeling joyful and proud", "happy"),
    ("Feeling so sad and depressed right now", "sad"),
    ("My heart is broken", "sad"),
    ("Everything went wrong today", "sad"),
    ("I feel lonely and upset", "sad"),
    ("Feeling peaceful and relaxed", "calm"),
    ("It is a quiet and serene day", "calm"),
    ("Meditating and resting", "calm"),
    ("Rain outside feels calming", "calm"),
    ("I am pumped and hyped up", "energetic"),
    ("Fired up and motivated", "energetic"),
    ("I feel intense and focused", "energetic"),
    ("Stressed but full of energy", "energetic"),
    ("I love you so much", "romantic"),
    ("Missing my partner", "romantic"),
    ("My crush is beautiful", "romantic"),
    ("Heart full of affection", "romantic"),
]

# PYTHON CONCEPTS: Pandas DataFrame (very simple explanation)
# Think of DataFrame like an Excel sheet in Python.
# Here we store training data in 2 columns:
# 1) text  -> example sentence
# 2) mood  -> correct label for that sentence
# This makes data easy to read and send to the ML model.
training_df = pd.DataFrame(TRAINING_TEXTS_AND_MOODS, columns=["text", "mood"])

# PYTHON CONCEPTS: Scikit-learn Pipeline and ML model training.
# Pipeline means: do steps in order automatically.
# Step 1: Converts words into numbers the computer can understand.
# Step 2: Naive Bayes learns patterns from those numbers to predict mood.
# .fit(...) is the training step where model learns from known examples.
MOOD_CLASSIFIER = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=200, lowercase=True)),
    ("clf", MultinomialNB()),
])
MOOD_CLASSIFIER.fit(training_df["text"], training_df["mood"])


def analyze_mood_with_ml(text):
    """Use scikit-learn model prediction with probability-based confidence."""
    prediction = MOOD_CLASSIFIER.predict([text])[0]
    confidence_scores = MOOD_CLASSIFIER.predict_proba([text])[0]
    class_names = MOOD_CLASSIFIER.named_steps["clf"].classes_
    max_confidence = float(np.max(confidence_scores))

    if max_confidence >= 0.7:
        confidence_level = "high"
    elif max_confidence >= 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return {
        "primary_mood": prediction,
        "confidence": confidence_level,
        "confidence_score": max_confidence,
        "all_scores": {m: float(s) for m, s in zip(class_names, confidence_scores)},
        "detected_emotions": [prediction],
        "explanation": f"ML model predicts {prediction} mood with {max_confidence:.1%} confidence.",
        "source": "ml_classifier",
    }


def analyze_mood_python(text):
    """Keyword-based fallback analyzer using dictionary/list logic."""
    normalized_text = text.lower()
    mood_scores = {mood: 0 for mood in MOOD_KEYWORDS.keys()}
    matched_keywords = []

    for mood, keywords in MOOD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_text:
                mood_scores[mood] += 1
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)

    ranked = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    primary_mood, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        confidence = "low"
        primary_mood = "calm"
    elif top_score >= 3:
        confidence = "high"
    elif top_score > second_score:
        confidence = "medium"
    else:
        confidence = "low"

    explanation = (
        f"I detected keywords indicating a {primary_mood} mood."
        if top_score > 0
        else "No strong mood keywords found, defaulting to calm."
    )

    return {
        "primary_mood": primary_mood,
        "confidence": confidence,
        "detected_emotions": [e.capitalize() for e in matched_keywords[:6]],
        "explanation": explanation,
        "source": "python_keyword_fallback",
    }


@app.route("/")
def index():
    return render_template("mood.html")


@app.route("/api/analyze-mood", methods=["POST"])
def api_analyze_mood():
    try:
        data = request.get_json() or {}
        user_text = (data.get("text") or "").strip()
        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        try:
            result = analyze_mood_with_ml(user_text)
        except Exception:
            result = analyze_mood_python(user_text)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/get-recommendations", methods=["POST"])
def api_get_recommendations():
    try:
        data = request.get_json() or {}
        mood = (data.get("mood") or "").lower().strip()
        if mood not in MOOD_KEYWORDS.keys():
            return jsonify({"error": "Invalid mood"}), 400

        recommendations = [song for song in MUSIC_DATABASE if song["mood"] == mood]
        return jsonify({"songs": recommendations, "count": len(recommendations)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/all-songs", methods=["GET"])
def api_all_songs():
    return jsonify({"songs": MUSIC_DATABASE, "total": len(MUSIC_DATABASE)})


if __name__ == "__main__":
    print("Mood-Based Music Recommender Backend (Python + Flask + Scikit-learn + Pandas)")
    print("Running on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
