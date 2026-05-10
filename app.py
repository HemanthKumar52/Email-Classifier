"""Flask web app: paste an email body, get a predicted category."""
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "model.joblib not found. Run `python train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    model = get_model()
    proba = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(
        ({"label": c, "confidence": float(p)} for c, p in zip(classes, proba)),
        key=lambda r: r["confidence"],
        reverse=True,
    )
    return jsonify({"prediction": ranked[0]["label"], "scores": ranked})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
