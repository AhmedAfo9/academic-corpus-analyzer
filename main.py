import json
import math
import os
import re
from collections import Counter

import numpy as np
import spacy
import textstat
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Academic Corpus Analyzer - Calibrated Engine v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    r"\bplays? a (?:pivotal|crucial) role\b", r"\bstands? as a\b",
    r"\bnavigat(?:e|es|ing) the complex", r"\bin the realm of\b",
    r"\bever-evolving\b", r"\ba myriad of\b", r"\bboasts? a\b",
    r"\bmeticulous(?:ly)?\b", r"\bseamless(?:ly)? integrat\w*\b",
    r"\bunlocks? the (?:potential|power)\b", r"\bharness(?:es|ing)? the power of\b",
    r"\bin today'?s (?:fast-paced|digital|ever-evolving) world\b",
    r"\bit is (?:important|crucial) to note that\b",
    r"\bcannot be overstated\b", r"\bserves? as a\b",
    r"\bpush(?:es|ing)? the boundaries\b", r"\binvaluable insights\b",
]

NEUTRAL_ACADEMIC_CONNECTORS = [
    r"\bmoreover\b", r"\bfurthermore\b", r"\bin other words\b",
    r"\bin conclusion\b", r"\bit is worth noting\b", r"\badditionally\b",
    r"\bhowever\b", r"\bthus\b", r"\bhence\b", r"\bconsequently\b",
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

def _regex_density(patterns, text_lower, total_words):
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text_lower))
    density = round((count / total_words) * 100, 3) if total_words else 0.0
    return count, density

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

def compute_pos_bigram_entropy(pos_tags):
    if len(pos_tags) < 2:
        return 0.0, 0.0
    bigrams = [f"{pos_tags[i]}_{pos_tags[i + 1]}" for i in range(len(pos_tags) - 1)]
    counts = Counter(bigrams)
    total = len(bigrams)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    unique = len(counts)
    max_entropy = math.log2(unique) if unique > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return round(entropy, 3), round(min(normalized, 1.0), 3)

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
        1 for token in doc
        if token.dep_ in ("nsubjpass", "auxpass", "nsubj:pass", "aux:pass")
    )
    passive_ratio = min(round((passive_instances / total_sentences) * 100, 2), 100.0)

    text_lower = text.lower()
    strong_count, strong_ai_cliche_density = _regex_density(
        STRONG_AI_CLICHE_PATTERNS, text_lower, total_words
    )
    weak_count, weak_connector_density = _regex_density(
        NEUTRAL_ACADEMIC_CONNECTORS, text_lower, total_words
    )

    awl_count = sum(1 for w in words if w in AWL_WORDS)
    awl_density = round((awl_count / total_words) * 100, 2)

    tree_depths = [get_sentence_depth(s) for s in sentences]
    avg_tree_depth = round(sum(tree_depths) / len(tree_depths), 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    pos_bigram_entropy, pos_bigram_entropy_norm = compute_pos_bigram_entropy(pos_tags)

    sentence_openers = [s[0].pos_ for s in sentences if len(s) > 0]
    sentence_opener_diversity = (
        round(len(set(sentence_openers)) / len(sentence_openers), 3)
        if sentence_openers else 0.0
    )

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
        "ai_words_count": strong_count,
        "strong_ai_cliche_count": strong_count,
        "strong_ai_cliche_density": strong_ai_cliche_density,
        "weak_connector_count": weak_count,
        "weak_connector_density": weak_connector_density,
        "awl_density": awl_density,
        "avg_tree_depth": avg_tree_depth,
        "pos_bigram_entropy": pos_bigram_entropy,
        "pos_bigram_entropy_norm": pos_bigram_entropy_norm,
        "sentence_opener_diversity": sentence_opener_diversity,
        "readability_grade": readability_grade,
    }

FEATURE_KEYS_FOR_MODEL = [
    "burstiness",
    "pos_bigram_entropy_norm",
    "sentence_opener_diversity",
    "hapax_ratio",
    "strong_ai_cliche_density",
    "weak_connector_density",
]

DEFAULT_FEATURE_MEAN = [4.2, 0.60, 0.58, 34.0, 0.05, 0.35]
DEFAULT_FEATURE_STD = [2.0, 0.13, 0.17, 8.0, 0.20, 0.35]

CLASS_WEIGHTS_DEFAULT = {
    "human": {
        "bias": 0.15,
        "burstiness_z": 0.9,
        "pos_bigram_entropy_norm_z": 0.5,
        "sentence_opener_diversity_z": 0.4,
        "hapax_ratio_z": 0.6,
        "strong_ai_cliche_density_z": -1.6,
        "weak_connector_density_z": 0.05,
    },
    "pure_ai": {
        "bias": -0.15,
        "burstiness_z": -1.1,
        "pos_bigram_entropy_norm_z": -0.7,
        "sentence_opener_diversity_z": -0.5,
        "hapax_ratio_z": -0.3,
        "strong_ai_cliche_density_z": 1.3,
        "weak_connector_density_z": 0.15,
    },
    "ai_humanized": {
        "bias": 0.0,
        "burstiness_z": -0.3,
        "pos_bigram_entropy_norm_z": -0.9,
        "sentence_opener_diversity_z": -0.2,
        "hapax_ratio_z": -0.5,
        "strong_ai_cliche_density_z": 0.5,
        "weak_connector_density_z": 0.05,
    },
}

CLASS_LABELS = {"human": "Human Baseline", "pure_ai": "Pure AI", "ai_humanized": "AI-Humanized"}

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")

def _load_trained_model():
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH) as f:
                data = json.load(f)
            if "class_weights" in data and "scaler" in data:
                return data["class_weights"], data["scaler"]
        except Exception:
            pass
    return None, None

def _get_scaler():
    _, trained_scaler = _load_trained_model()
    if trained_scaler is not None:
        return trained_scaler["mean"], trained_scaler["scale"]
    return DEFAULT_FEATURE_MEAN, DEFAULT_FEATURE_STD

def _get_class_weights():
    trained_weights, _ = _load_trained_model()
    return trained_weights if trained_weights is not None else CLASS_WEIGHTS_DEFAULT

def _compute_zscores(m):
    means, stds = _get_scaler()
    zs = {}
    for i, key in enumerate(FEATURE_KEYS_FOR_MODEL):
        std = stds[i] if stds[i] else 1e-6
        z = (m[key] - means[i]) / std
        zs[f"{key}_z"] = max(-3.0, min(3.0, z))
    return zs

def _softmax(logits: dict):
    values = list(logits.values())
    top = max(values)
    exps = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: round((v / total) * 100, 1) for k, v in exps.items()}

def _feature_vector(res):
    means, stds = _get_scaler()
    return np.array([
        (res[k] - means[i]) / (stds[i] if stds[i] else 1e-6)
        for i, k in enumerate(FEATURE_KEYS_FOR_MODEL)
    ])

def classify_single_text_logic(m):
    z = _compute_zscores(m)
    weights = _get_class_weights()

    logits = {}
    for cls, w in weights.items():
        logit = w.get("bias", 0.0)
        for feat_key, z_val in z.items():
            logit += w.get(feat_key, 0.0) * z_val
        logits[cls] = logit

    probs = _softmax(logits)
    predicted_key = max(probs, key=probs.get)
    predicted_class = CLASS_LABELS.get(predicted_key, predicted_key)

    sorted_probs = sorted(probs.values(), reverse=True)
    gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 100.0
    if gap >= 25:
        confidence = "high"
    elif gap >= 10:
        confidence = "moderate"
    else:
        confidence = "low"

    flags = []
    if m["words"] < 150:
        flags.append(
            f"Short text ({m['words']} words): stylometric estimates (burstiness, entropy, "
            f"hapax ratio) get noisier below roughly 150 words. Treat this result cautiously."
        )
    if z.get("strong_ai_cliche_density_z", 0) > 1.0:
        flags.append(
            f"Elevated density of LLM-associated phrasing ({m['strong_ai_cliche_density']} "
            f"per 100 words) — vocabulary disproportionately common in raw or lightly-edited "
            f"AI output."
        )
    if z.get("pos_bigram_entropy_norm_z", 0) < -1.0:
        flags.append(
            "Below-typical POS-transition entropy: sentence structures are more templated/"
            "repetitive than typical academic writing. This pattern often survives word-level "
            "paraphrasing, which is why it carries weight toward AI-humanized."
        )
    if z.get("burstiness_z", 0) < -1.0:
        flags.append(
            "Low sentence-length variation (burstiness): human writing typically shows more "
            "rhythmic variation between sentences."
        )
    if m["weak_connector_density"] > 0 and z.get("strong_ai_cliche_density_z", 0) < 0.5:
        flags.append(
            "Standard academic transitions present (e.g. moreover/furthermore/however) without "
            "accompanying AI-associated phrasing — consistent with expected academic register "
            "and not treated as evidence of AI authorship."
        )
    if confidence == "low":
        flags.append(
            "Signals are mixed: the top two classes are close in probability. Treat this as "
            "inconclusive rather than a confident determination."
        )

    lexical_score = round(min(100.0, max(0.0, (m["guiraud_r"] * 10) + (m["lexical_density"] * 0.8))), 1)
    syntactic_score = round(min(100.0, max(0.0, (m["avg_tree_depth"] * 12) + (m["mls"] * 1.2))), 1)
    entropy_score = round(min(100.0, max(0.0, m["pos_bigram_entropy_norm"] * 100)), 1)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "human": probs.get("human", 0.0),
            "pure_ai": probs.get("pure_ai", 0.0),
            "ai_humanized": probs.get("ai_humanized", 0.0),
        },
        "sub_scores": {
            "lexical_authenticity": lexical_score,
            "syntactic_complexity": syntactic_score,
            "stylistic_entropy": entropy_score,
        },
        "diagnostic_flags": flags,
        "disclaimer": (
            "Statistical indicator, not proof of authorship. Rule-based and lightly-calibrated "
            "detectors misfire on genuine human writing — especially L2/EFL academic writing — "
            "at non-trivial rates. Do not use this output as sole evidence in an academic "
            "misconduct decision; pair it with human judgment and, ideally, a conversation with "
            "the writer."
        ),
    }

def compute_residual_footprint(human_res, ai_res, humanized_res):
    v_human = _feature_vector(human_res)
    v_ai = _feature_vector(ai_res)
    v_hum = _feature_vector(humanized_res)

    dist_to_ai = float(np.linalg.norm(v_hum - v_ai))
    dist_to_human = float(np.linalg.norm(v_hum - v_human))
    total = dist_to_ai + dist_to_human
    if total == 0:
        return 50.0
    return round((dist_to_human / total) * 100, 1)

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer Calibrated API v4 is Live"}

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

    residual_footprint = None
    if ai_res and humanized_res and human_res:
        residual_footprint = compute_residual_footprint(human_res, ai_res, humanized_res)

    return {
        "metrics": {
            "human": human_res,
            "pure_ai": ai_res,
            "ai_humanized": humanized_res,
        },
        "residual_ai_footprint_percentage": residual_footprint,
    }

@app.post("/analyze-single")
def analyze_single_text(data: SingleInput):
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = analyze_single_corpus(data.text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    classification = classify_single_text_logic(metrics)

    return {
        "metrics": metrics,
        "classification": classification,
    }
