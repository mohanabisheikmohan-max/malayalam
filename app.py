import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__)

MODEL_NAME = "gemini-3.1-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in .env")

client = genai.Client(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "chatbot_config", "r", encoding="utf-8") as config_file:
    SYSTEM_PROMPT = config_file.read().strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a Malayalam study question."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config={"system_instruction": SYSTEM_PROMPT},
        )

        return jsonify({
            "reply": response.text or "I could not generate a response."
        })

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "Unable to process the request right now."
        }), 500


if __name__ == "__main__":
    app.run()
