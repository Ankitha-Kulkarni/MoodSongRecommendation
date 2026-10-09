"""Train the mood classifier.

Run directly (`python train_model.py`) to save `mood_model.joblib`.
The Streamlit app also imports `train()` and trains in memory if the saved
file is missing or was built with an incompatible scikit-learn version.
"""
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent / "mood_model.joblib"

DATA = {
    "happy": [
        "I feel happy", "I feel amazing today", "I am so excited",
        "I feel joyful", "life is great right now", "I just got promoted",
        "I am in a really good mood", "everything is going well",
        "I feel on top of the world", "I am so grateful today",
        "I feel cheerful and positive", "today is a wonderful day",
        "I am thrilled about the news", "I feel proud of myself",
        "I am smiling for no reason", "feeling fantastic and full of joy",
    ],
    "sad": [
        "I am sad", "I feel depressed", "I feel lonely", "I am crying",
        "my heart is broken", "I miss someone badly", "I feel empty inside",
        "I feel down and hopeless", "I had a terrible day and feel low",
        "I am heartbroken", "nothing feels good anymore",
        "I feel like crying tonight", "I am feeling blue",
        "I want something emotional", "I feel so alone and tired of everything",
        "I feel sorrow and grief",
    ],
    "angry": [
        "I feel angry", "I am frustrated", "I feel furious",
        "I am so annoyed right now", "I am sick of everyone",
        "I want to scream", "I am really mad", "I hate this so much",
        "my boss made me so irritated", "I feel rage building up",
        "I am fed up with everything", "I am pissed off",
        "I need to let out my anger", "I feel aggressive and bitter",
        "everything is making me furious", "I am done with this nonsense",
    ],
    "calm": [
        "I want peace", "I feel calm", "I want to relax",
        "I need to unwind after work", "I feel peaceful and quiet",
        "I want to meditate", "I am trying to chill", "I feel serene",
        "I want something soothing for sleep", "I want a quiet evening",
        "I am relaxing at home with tea", "I need to destress",
        "I want gentle background music for studying", "I feel at ease",
        "I want to slow down and breathe", "a lazy rainy afternoon",
    ],
    "romantic": [
        "I am romantic", "I am in love", "I miss my girlfriend",
        "I have a crush on someone", "I want music for a date night",
        "I feel so loved", "I am thinking about my partner",
        "I want a love song for my husband", "it is our anniversary",
        "I feel affectionate", "I am falling for someone",
        "I want to dedicate a song to my wife", "candlelight dinner with my love",
        "I am head over heels", "I feel passion and romance",
        "I want to propose to her",
    ],
    "energetic": [
        "I feel energetic", "I want workout music", "I am going to the gym",
        "I need to get pumped up", "I want to run faster",
        "I feel full of energy", "I need motivation to hit my goals",
        "I want to dance all night", "I am ready for a party",
        "I want a high energy playlist", "time to train hard",
        "I need hype music before the game", "I feel unstoppable",
        "I want to go crazy and jump around", "I am hyper and wide awake",
        "I need beats for my morning run",
    ],
}


def build_pipeline():
    return Pipeline([
        # Character n-grams cope with typos and word forms (angry/angrier)
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def get_xy():
    texts, labels = [], []
    for mood, sentences in DATA.items():
        texts.extend(sentences)
        labels.extend([mood] * len(sentences))
    return texts, labels


def train(evaluate=False):
    texts, labels = get_xy()
    pipeline = build_pipeline().fit(texts, labels)
    if not evaluate:
        return pipeline
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(build_pipeline(), texts, labels, cv=cv)
    return pipeline, scores


def main():
    pipeline, scores = train(evaluate=True)
    joblib.dump({"pipeline": pipeline, "sklearn": sklearn.__version__}, MODEL_PATH)
    print(f"Model trained on {len(get_xy()[0])} sentences, "
          f"{len(DATA)} moods (scikit-learn {sklearn.__version__}).")
    print(f"Cross-validated accuracy: {scores.mean():.0%} (+/- {scores.std():.0%})")
    print(f"Saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    sys.exit(main())