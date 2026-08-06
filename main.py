import json
import math
import os
import re
from collections import Counter

import numpy as np
import spacy
import textstat
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Academic Corpus Analyzer - Hybrid AI Engine v5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") # ضع مفتاحك هنا أو كمتغير بيئة في Render
OPENROUTER_MODEL = "openai/gpt-4o-mini" # نموذج سريع، دقيق، ورخيص جداً

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

STRONG_AI_CLICHE_PATTERNS = [
    r"\bdelves?\b", r"\bdelving\b", r"\btapestry\b", r"\bmultifaceted\b",
    r"\btestament to\b", r"\bunderscores?\b", r"\bunderscoring\b",
    r"\binterplay\b", r"\bfoster(?:s|ing)? a\b", r"\bvibrant\b",
    r"\bplays? a (?:pivotal|crucial) role\b", r"\bstands? as a\b"
]

class SingleInput(BaseModel):
    text: str

def analyze_stylometrics(text: str):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]
    total_words = len(words)
    total_sentences = len(sentences)

    if total_words < 5 or total_sentences == 0:
        return None

    unique_words = len(set(words))
    guiraud_r = round(float(unique_words / np.sqrt(total_words)), 2)
    sentence_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    burstiness = round(float(np.std(sentence_lengths)), 2) if len(sentence_lengths) > 1 else 0.0

    content_words = [token for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lexical_density = round((len(content_words) / total_words) * 100, 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    counts = Counter(bigrams)
    total_b = len(bigrams)
    probs = [c / total_b for c in counts.values()] if total_b > 0 else []
    entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    norm_entropy = round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0

    return {
        "words": total_words,
        "sentences": total_sentences,
        "guiraud_r": guiraud_r,
        "lexical_density": lexical_density,
        "burstiness": burstiness,
        "pos_entropy_norm": norm_entropy,
        "mls": round(total_words / total_sentences, 2)
    }

async def call_openrouter_detector(text: str):
    if not OPENROUTER_API_KEY:
        return None

    system_prompt = (
        "You are an expert Forensic Stylometric Linguist specializing in detecting AI-generated text, "
        "human writing, and AI paraphrasing (humanized text).\n"
        "Analyze the provided text carefully for underlying LLM predictability, syntactic uniformness, "
        "semantic density, and subtle paraphrasing footprints.\n"
        "Return ONLY a JSON object with the following keys:\n"
        "{\n"
        '  "human_prob": float (0-100),\n'
        '  "pure_ai_prob": float (0-100),\n'
        '  "ai_humanized_prob": float (0-100),\n'
        '  "verdict": string ("Human Baseline" | "Pure AI" | "AI-Humanized"),\n'
        '  "confidence": string ("high" | "moderate" | "low"),\n'
        '  "reasoning": string (concise explanation of linguistic evidence)\n'
        "}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
    except Exception as e:
        print(f"OpenRouter Error: {e}")
    return None

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Hybrid v5 Engine is Live"}

@app.post("/analyze-single")
async def analyze_single_text(data: SingleInput):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = analyze_stylometrics(data.text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    # Call OpenRouter LLM Classifier
    ai_evaluation = await call_openrouter_detector(data.text)

    if ai_evaluation:
        classification = {
            "predicted_class": ai_evaluation.get("verdict", "Human Baseline"),
            "confidence": ai_evaluation.get("confidence", "moderate"),
            "probabilities": {
                "human": ai_evaluation.get("human_prob", 33.3),
                "pure_ai": ai_evaluation.get("pure_ai_prob", 33.3),
                "ai_humanized": ai_evaluation.get("ai_humanized_prob", 33.3)
            },
            "sub_scores": {
                "lexical_authenticity": round(metrics["guiraud_r"] * 10, 1),
                "syntactic_complexity": round(metrics["mls"] * 1.5, 1),
                "stylistic_entropy": round(metrics["pos_entropy_norm"] * 100, 1)
            },
            "diagnostic_flags": [ai_evaluation.get("reasoning", "Linguistic pattern evaluated via OpenRouter.")],
            "disclaimer": "Hybrid evaluation powered by OpenRouter LLM & spaCy Stylometrics."
        }
    else:
        # Fallback if API key is missing or fails
        classification = {
            "predicted_class": "Human Baseline",
            "confidence": "low",
            "probabilities": {"human": 100.0, "pure_ai": 0.0, "ai_humanized": 0.0},
            "sub_scores": {"lexical_authenticity": 50, "syntactic_complexity": 50, "stylistic_entropy": 50},
            "diagnostic_flags": ["API Key missing or unreachable. Standard fallback applied."],
            "disclaimer": "Fallback mode active."
        }

    return {
        "metrics": metrics,
        "classification": classification
    }
