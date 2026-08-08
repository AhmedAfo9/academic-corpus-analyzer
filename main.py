import os
import re
import numpy as np
import spacy
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

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

class CorpusInput(BaseModel):
    corpus_a: Optional[str] = ""
    corpus_b: Optional[str] = ""
    corpus_c: Optional[str] = ""
    text_a: Optional[str] = ""
    text_b: Optional[str] = ""
    text_c: Optional[str] = ""
    text: Optional[str] = ""

def analyze_single_corpus(text: str):
    if not text or not text.strip():
        return None

    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]

    total_words = len(words)
    total_sentences = len(sentences)

    if total_words < 5 or total_sentences == 0:
        return None

    unique_words = len(set(words))
    guiraud_r = round(float(unique_words / np.sqrt(total_words)), 2)
    mls = round(total_words / total_sentences, 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    pos_transition_ratio = round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.0

    return {
        "words": total_words,
        "sentences": total_sentences,
        "guiraud_r": guiraud_r,
        "mls": mls,
        "pos_transition_ratio": pos_transition_ratio,
    }

async def query_modal_editlens(text: str):
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            res = await client.post(MODAL_EDITLENS_URL, json={"text": text})
            if res.status_code == 200:
                data = res.json()
                if "ai_probability" in data:
                    return round(data["ai_probability"] * 100, 1), None
                probs = data.get("probs", [])
                if len(probs) >= 2:
                    return round(probs[1] * 100, 1), None
                elif len(probs) == 1:
                    return round(probs[0] * 100, 1), None
            return None, f"Status code: {res.status_code}"
        except Exception as e:
            return None, str(e)

def build_response_payload(metrics, ai_score, err):
    flags = []
    if ai_score is not None:
        if ai_score >= 50.0:
            predicted_class = "Pure AI"
        elif ai_score >= 20.0:
            predicted_class = "AI-Humanized"
        else:
            predicted_class = "Human Baseline"

        confidence = "high" if abs(ai_score - 50.0) > 20.0 else "moderate"
        flags.append(f"Official Neural Engine: {ai_score}% AI probability footprint.")
    else:
        flags.append(f"Modal Engine Notice: {err or 'Fallback active'}")
        ai_score = 15.0
        predicted_class = "Human Baseline"
        confidence = "low"

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
        "diagnostic_flags": flags,
        "disclaimer": "Official EditLens Neural Engine."
    }

    return {
        "metrics": metrics,
        "classification": classification,
    }

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Active"}

@app.post("/analyze-single")
async def analyze_single_text(data: SingleInput):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = analyze_single_corpus(data.text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    ai_score, err = await query_modal_editlens(data.text)
    return build_response_payload(metrics, ai_score, err)

@app.post("/analyze")
@app.post("/analyze-corpus")
async def analyze_corpora(data: CorpusInput):
    c_a = data.corpus_a or data.text_a or data.text
    c_b = data.corpus_b or data.text_b
    c_c = data.corpus_c or data.text_c

    results = {}
    for key, text_val in [("corpus_a", c_a), ("corpus_b", c_b), ("corpus_c", c_c)]:
        if text_val and text_val.strip():
            m = analyze_single_corpus(text_val)
            if m:
                ai_s, err_msg = await query_modal_editlens(text_val)
                results[key] = build_response_payload(m, ai_s, err_msg)

    if not results:
        raise HTTPException(status_code=400, detail="No valid text provided in corpora.")

    return {
        "status": "success",
        "results": results,
        **results
    }
