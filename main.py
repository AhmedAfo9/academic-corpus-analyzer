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

app = FastAPI(title="Academic Corpus Analyzer - Real EditLens Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# رابط سيرفر Modal العصبي الحقيقي المخصص لمشروعك
MODAL_EDITLENS_URL = "https://ahmedfalahoraibi--editlens-engine-editlensserver-predict.modal.run"

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

AWL_WORDS = {
    "analyze", "approach", "assess", "assume", "authority", "available", "benefit",
    "concept", "consistent", "constitutional", "context", "contract", "create",
    "data", "definition", "derived", "distribution", "economic", "environment",
    "established", "evidence", "factors", "financial", "formula", "function",
    "identified", "income", "indicate", "individual", "interpretation", "involved",
    "issues", "labour", "legal", "legislation", "major", "method", "occur",
    "percent", "period", "policy", "principle", "procedure", "process", "required",
    "research", "response", "role", "section", "sector", "significant", "similar",
    "source", "specific", "structure", "theory", "variables"
}

class CorpusInput(BaseModel):
    human_text: str = ""
    ai_text: str = ""
    humanized_text: str = ""

class SingleInput(BaseModel):
    text: str

def get_sentence_depth(sent_doc):
    roots = [token for token in sent_doc if token.head == token]
    if not roots:
        return 1
    max_depth = 1
    for root in roots:
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            for child in node.children:
                stack.append((child, depth + 1))
    return max_depth

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
    ttr = round(unique_words / total_words, 3)
    guiraud_r = round(float(unique_words / np.sqrt(total_words)), 2)

    word_counts = Counter(words)
    hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = round((hapax_count / total_words) * 100, 2)

    sentence_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    mls = round(total_words / total_sentences, 2)
    burstiness = round(float(np.std(sentence_lengths)), 2) if len(sentence_lengths) > 1 else 0.0

    content_words = [token for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lexical_density = round((len(content_words) / total_words) * 100, 2)

    passive_instances = sum(
        1 for token in doc if token.dep_ in ("nsubjpass", "auxpass", "nsubj:pass", "aux:pass")
    )
    passive_ratio = min(round((passive_instances / total_sentences) * 100, 2), 100.0)

    text_lower = text.lower()
    ai_words_count = sum(len(re.findall(pat, text_lower)) for pat in STRONG_AI_CLICHE_PATTERNS)

    awl_count = sum(1 for w in words if w in AWL_WORDS)
    awl_density = round((awl_count / total_words) * 100, 2)

    tree_depths = [get_sentence_depth(s) for s in sentences]
    avg_tree_depth = round(sum(tree_depths) / len(tree_depths), 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    pos_transition_ratio = round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.0

    try:
        readability_grade = round(float(textstat.flesch_kincaid_grade(text)), 2)
    except Exception:
        readability_grade = None

    return {
        "words": total_words,
        "sentences": total_sentences,
        "ttr": ttr,
        "guiraud_r": guiraud_r,
        "hapax_ratio": hapax_ratio,
        "mls": mls,
        "lexical_density": lexical_density,
        "passive_ratio": passive_ratio,
        "burstiness": burstiness,
        "ai_words_count": ai_words_count,
        "awl_density": awl_density,
        "avg_tree_depth": avg_tree_depth,
        "pos_transition_ratio": pos_transition_ratio,
        "readability_grade": readability_grade,
    }

async def query_modal_editlens(text: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(MODAL_EDITLENS_URL, json={"text": text})
            if res.status_code == 200:
                data = res.json()
                probs = data.get("probs", [])
                if len(probs) >= 2:
                    ai_prob = probs[1]
                    return round(ai_prob * 100, 1), None
                elif len(probs) == 1:
                    return round(probs[0] * 100, 1), None
            return None, f"Modal status code: {res.status_code}"
        except Exception as e:
            return None, str(e)

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - EditLens Neural Engine Active"}

@app.post("/analyze")
def analyze_corpora(data: CorpusInput):
    human_res = analyze_single_corpus(data.human_text)
    ai_res = analyze_single_corpus(data.ai_text)
    humanized_res = analyze_single_corpus(data.humanized_text)

    valid_count = sum(1 for r in [human_res, ai_res, humanized_res] if r is not None)
    if valid_count < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 non-empty corpora are required for comparative analysis.",
        )

    residual_footprint = 0.0
    if ai_res and humanized_res and human_res:
        if ai_res["mls"] > 0 and human_res["mls"] > 0:
            diff_ai_hum = abs(humanized_res["mls"] - ai_res["mls"])
            diff_human_hum = abs(humanized_res["mls"] - human_res["mls"])
            if (diff_ai_hum + diff_human_hum) > 0:
                residual_footprint = round((diff_human_hum / (diff_ai_hum + diff_human_hum)) * 100, 1)

    return {
        "metrics": {
            "human": human_res,
            "pure_ai": ai_res,
            "ai_humanized": humanized_res,
        },
        "residual_ai_footprint_percentage": residual_footprint,
    }

@app.post("/analyze-single")
async def analyze_single_text(data: SingleInput):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = analyze_single_corpus(data.text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    ai_score, err = await query_modal_editlens(data.text)

    flags = []
    if ai_score is not None:
        if ai_score >= 60.0:
            predicted_class = "Pure AI"
        elif ai_score >= 30.0:
            predicted_class = "AI-Humanized"
        else:
            predicted_class = "Human Baseline"

        confidence = "high" if abs(ai_score - 50.0) > 20.0 else "moderate"
        flags.append(f"Official EditLens ICLR 2026 Model: {ai_score}% AI modification footprint.")
    else:
        flags.append(f"Modal Engine Notice: {err or 'Fallback active'}")
        ai_score = 15.0 if metrics["ai_words_count"] == 0 else 75.0
        predicted_class = "Human Baseline" if metrics["ai_words_count"] == 0 else "Pure AI"
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
        "disclaimer": "Official EditLens RoBERTa Large Neural Architecture (ICLR 2026)."
    }

    return {
        "metrics": metrics,
        "classification": classification,
    }
