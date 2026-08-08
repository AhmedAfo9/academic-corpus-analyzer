import os
import re
import math
import asyncio
import numpy as np
import spacy
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Academic Corpus Analyzer Engine")

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

# قائمة الكلمات الأكاديمية الشائعة (AWL Sample)
AWL_KEYWORDS = {
    "analysis", "approach", "area", "assessment", "assume", "authority", "available", "benefit",
    "concept", "consistent", "constitutional", "context", "contract", "create", "data", "definition",
    "derived", "distribution", "economic", "environment", "established", "estimate", "evidence",
    "factors", "financial", "formula", "function", "identified", "income", "indicate", "individual",
    "interpretation", "involved", "issues", "labour", "legal", "legislation", "major", "method",
    "percent", "period", "policy", "principle", "procedure", "process", "required", "research",
    "response", "role", "section", "sector", "significant", "similar", "source", "specific",
    "structure", "theory", "variables", "academic", "fundamental", "empirical", "methodology"
}

# كلمات الذكاء الاصطناعي الشائعة (AI Buzzwords)
AI_BUZZWORDS = {
    "delve", "realm", "tapestry", "testament", "pivotal", "underscore", "crucial", "multifaceted",
    "interplay", "beacon", "paramount", "fostering", "harnessing", "unwavering", "vibrant",
    "holistic", "seamless", "synergy", "paradigm", "transformative", "elucidate"
}

def compute_full_metrics(text: str):
    if not text or not text.strip():
        return None

    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]

    total_words = len(words)
    total_sentences = len(sentences)

    if total_words < 3 or total_sentences == 0:
        return None

    unique_words = len(set(words))
    
    # 1. TTR
    ttr = round(unique_words / total_words, 3)
    
    # 2. Guiraud's R Index
    guiraud_r = round(float(unique_words / np.sqrt(total_words)), 2)
    
    # 3. Hapax Legomena Ratio (%)
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = round((hapax_count / total_words) * 100, 1)

    # 4. MLS
    mls = round(total_words / total_sentences, 2)

    # 5. Lexical Density (%)
    content_words = [token for token in doc if token.is_alpha and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
    lexical_density = round((len(content_words) / total_words) * 100, 1)

    # 6. Passive Voice Ratio (%)
    passive_count = 0
    for token in doc:
        if token.dep_ in {"passive", "auxpass"}:
            passive_count += 1
    passive_ratio = round((passive_count / total_sentences) * 100, 1)

    # 7. AWL Density (%)
    awl_count = sum(1 for w in words if w in AWL_KEYWORDS)
    awl_density = round((awl_count / total_words) * 100, 1)

    # 8. Avg Dependency Tree Depth
    def get_depth(node):
        if not list(node.children):
            return 1
        return 1 + max(get_depth(child) for child in node.children)

    tree_depths = [get_depth(sent.root) for sent in doc.sents]
    avg_tree_depth = round(float(np.mean(tree_depths)), 1) if tree_depths else 1.0

    # 9. POS Transition Ratio (Entropy)
    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    pos_transition_ratio = round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.0

    # 10. Readability Grade Level (Flesch-Kincaid Estimate)
    syllables = sum(len(re.findall(r'[aeiouy]+', w)) for w in words)
    readability_grade = round(0.39 * (total_words / total_sentences) + 11.8 * (syllables / total_words) - 15.59, 1)
    readability_grade = max(1.0, min(20.0, readability_grade))

    # 11. Burstiness (Sentence Length Variance)
    sent_lengths = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]
    burstiness = round(float(np.std(sent_lengths)), 2) if len(sent_lengths) > 1 else 0.0

    # 12. AI Buzzwords Count
    ai_words_count = sum(1 for w in words if w in AI_BUZZWORDS)

    return {
        "words": total_words,
        "sentences": total_sentences,
        "ttr": ttr,
        "guiraud_r": guiraud_r,
        "hapax_ratio": hapax_ratio,
        "mls": mls,
        "lexical_density": lexical_density,
        "passive_ratio": passive_ratio,
        "awl_density": awl_density,
        "avg_tree_depth": avg_tree_depth,
        "pos_transition_ratio": pos_transition_ratio,
        "readability_grade": readability_grade,
        "burstiness": burstiness,
        "ai_words_count": ai_words_count
    }

async def query_modal_editlens(text: str):
    if not text or not text.strip():
        return 15.0, "Empty text"
        
    async with httpx.AsyncClient(timeout=30.0) as client:
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
            return 15.0, f"Status code: {res.status_code}"
        except Exception as e:
            return 15.0, str(e)

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Fully Synchronized"}

@app.post("/analyze-single")
async def analyze_single_text(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    text = data.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = compute_full_metrics(text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    ai_score, err = await query_modal_editlens(text)

    if ai_score >= 50.0:
        predicted_class = "Pure AI"
    elif ai_score >= 20.0:
        predicted_class = "AI-Humanized"
    else:
        predicted_class = "Human Baseline"

    human_prob = round(max(0.0, 100.0 - ai_score), 1)
    pure_ai_prob = round(ai_score if predicted_class == "Pure AI" else ai_score * 0.5, 1)
    humanized_prob = round(ai_score if predicted_class == "AI-Humanized" else max(0.0, 100.0 - abs(50.0 - ai_score) * 2), 1)

    return {
        "metrics": metrics,
        "classification": {
            "predicted_class": predicted_class,
            "probabilities": {
                "human": human_prob,
                "pure_ai": pure_ai_prob,
                "ai_humanized": humanized_prob
            },
            "sub_scores": {
                "lexical_authenticity": round(min(100.0, metrics["guiraud_r"] * 12), 1),
                "syntactic_complexity": round(min(100.0, metrics["mls"] * 2.5), 1),
                "stylistic_entropy": round(min(100.0, metrics["pos_transition_ratio"] * 100), 1)
            },
            "diagnostic_flags": [
                f"Neural Engine Evaluation: {ai_score}% AI probability footprint.",
                f"Lexical Diversity Index (TTR): {metrics['ttr']}",
                f"Syntactic Complexity (MLS): {metrics['mls']} words/sentence."
            ]
        }
    }

@app.post("/analyze")
@app.post("/analyze-corpus")
async def analyze_corpora(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    human_text = data.get("human_text", "")
    ai_text = data.get("ai_text", "")
    humanized_text = data.get("humanized_text", "")

    inputs = [
        ("human", human_text),
        ("pure_ai", ai_text),
        ("ai_humanized", humanized_text)
    ]

    active_tasks = []
    keys = []
    texts = []

    for key, txt in inputs:
        if txt and txt.strip():
            keys.append(key)
            texts.append(txt)
            active_tasks.append(query_modal_editlens(txt))

    if not active_tasks:
        raise HTTPException(status_code=400, detail="Please provide text in at least one corpus.")

    ai_scores = await asyncio.gather(*active_tasks)

    metrics_result = {}
    score_map = {}

    for key, txt, (score, err) in zip(keys, texts, ai_scores):
        m = compute_full_metrics(txt)
        if m:
            metrics_result[key] = m
            score_map[key] = score

    score_ai = score_map.get("pure_ai", 0.0)
    score_hum = score_map.get("ai_humanized", 0.0)
    residual_footprint = round(abs(score_ai - score_hum), 1) if ("pure_ai" in score_map and "ai_humanized" in score_map) else round(score_hum, 1)

    return {
        "status": "success",
        "residual_ai_footprint_percentage": residual_footprint,
        "metrics": metrics_result
    }
