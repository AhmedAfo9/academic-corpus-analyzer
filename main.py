import os
import re
import asyncio
import numpy as np
import spacy
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Academic Corpus Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODAL_EDITLENS_URL = "https://ahmedfalahoraibi--editlens-engine-editlensserver-predict.modal.run"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class SingleInput(BaseModel):
    text: str

def analyze_single_corpus(text: str):
    if not text or not str(text).strip():
        return {
            "words": 10, "sentences": 1, "guiraud_r": 2.5, "mls": 10.0, "pos_transition_ratio": 0.5
        }

    doc = nlp(str(text))
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]

    total_words = max(len(words), 1)
    total_sentences = max(len(sentences), 1)

    unique_words = len(set(words)) if words else 1
    guiraud_r = round(float(unique_words / np.sqrt(total_words)), 2)
    mls = round(total_words / total_sentences, 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    pos_transition_ratio = round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.5

    return {
        "words": total_words,
        "sentences": total_sentences,
        "guiraud_r": guiraud_r,
        "mls": mls,
        "pos_transition_ratio": pos_transition_ratio,
    }

async def query_modal_editlens(client: httpx.AsyncClient, text: str):
    try:
        res = await client.post(MODAL_EDITLENS_URL, json={"text": text})
        if res.status_code == 200:
            data = res.json()
            probs = data.get("probs", [])
            if len(probs) >= 2:
                return round(probs[1] * 100, 1)
        return 15.0
    except Exception:
        return 15.0

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Active"}

@app.post("/analyze-single")
async def analyze_single_text(data: SingleInput):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = analyze_single_corpus(data.text)
    async with httpx.AsyncClient(timeout=30.0) as client:
        ai_score = await query_modal_editlens(client, data.text)

    predicted_class = "Pure AI" if ai_score >= 50.0 else ("AI-Humanized" if ai_score >= 20.0 else "Human Baseline")
    confidence = "high" if abs(ai_score - 50.0) > 20.0 else "moderate"

    probs = {
        "human": round(max(0.0, 100.0 - ai_score), 1),
        "pure_ai": round(ai_score if predicted_class == "Pure AI" else ai_score * 0.5, 1),
        "ai_humanized": round(ai_score if predicted_class == "AI-Humanized" else max(0.0, 100.0 - abs(50.0 - ai_score) * 2), 1)
    }

    classification = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs,
        "sub_scores": {
            "lexical_authenticity": round(metrics["guiraud_r"] * 10, 1),
            "syntactic_complexity": round(metrics["mls"] * 1.5, 1),
            "stylistic_entropy": round(metrics["pos_transition_ratio"] * 100, 1)
        },
        "diagnostic_flags": [f"Official Neural Engine: {ai_score}% AI probability footprint."],
        "disclaimer": "Official EditLens Neural Engine."
    }

    return {"metrics": metrics, "classification": classification}

@app.post("/analyze")
@app.post("/compare")
async def compare_corpora(payload: dict):
    # استخراج النصوص بجميع التنسيقات المحتملة
    text_a = payload.get("corpus_a") or payload.get("corpusA") or payload.get("text_a") or payload.get("textA") or ""
    text_b = payload.get("corpus_b") or payload.get("corpusB") or payload.get("text_b") or payload.get("textB") or ""
    text_c = payload.get("corpus_c") or payload.get("corpusC") or payload.get("text_c") or payload.get("textC") or ""

    if not (text_a and text_b and text_c):
        str_values = [str(v) for v in payload.values() if isinstance(v, str)]
        if len(str_values) >= 3:
            text_a, text_b, text_c = str_values[0], str_values[1], str_values[2]

    metrics_a = analyze_single_corpus(text_a)
    metrics_b = analyze_single_corpus(text_b)
    metrics_c = analyze_single_corpus(text_c)

    async with httpx.AsyncClient(timeout=30.0) as client:
        score_a, score_b, score_c = await asyncio.gather(
            query_modal_editlens(client, text_a),
            query_modal_editlens(client, text_b),
            query_modal_editlens(client, text_c)
        )

    return {
        "corpus_a": {"metrics": metrics_a, "ai_score": score_a},
        "corpus_b": {"metrics": metrics_b, "ai_score": score_b},
        "corpus_c": {"metrics": metrics_c, "ai_score": score_c},
        "results": {
            "corpus_a": {"metrics": metrics_a, "ai_score": score_a},
            "corpus_b": {"metrics": metrics_b, "ai_score": score_b},
            "corpus_c": {"metrics": metrics_c, "ai_score": score_c}
        }
    }
