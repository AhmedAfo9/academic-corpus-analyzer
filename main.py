import os
import re
import math
import asyncio
import numpy as np
import spacy
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

app = FastAPI(
    title="Academic Corpus Analyzer - Research Engine",
    description="Multidimensional Linguistic & Stylometric Analysis Platform"
)

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

FUNCTION_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}

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

AI_BUZZWORDS = {
    "delve", "realm", "tapestry", "testament", "pivotal", "underscore", "crucial", "multifaceted",
    "interplay", "beacon", "paramount", "fostering", "harnessing", "unwavering", "vibrant",
    "holistic", "seamless", "synergy", "paradigm", "transformative", "elucidate"
}

def compute_mtld(tokens: List[str], threshold: float = 0.72) -> float:
    if len(tokens) < 10:
        return 0.0

    def evaluate_factor_count(token_list):
        factors = 0.0
        current_types = set()
        token_count = 0

        for token in token_list:
            token_count += 1
            current_types.add(token)
            ttr = len(current_types) / token_count
            if ttr <= threshold:
                factors += 1.0
                current_types = set()
                token_count = 0

        if token_count > 0:
            current_ttr = len(current_types) / token_count
            if current_ttr < 1.0:
                factors += (1.0 - current_ttr) / (1.0 - threshold)

        return factors if factors > 0 else 1.0

    forward = evaluate_factor_count(tokens)
    backward = evaluate_factor_count(list(reversed(tokens)))
    
    mtld_val = (len(tokens) / forward + len(tokens) / backward) / 2.0
    return round(float(mtld_val), 2)

def compute_mattr(tokens: List[str], window_size: int = 50) -> float:
    if len(tokens) < window_size:
        return round(len(set(tokens)) / max(1, len(tokens)), 3)
    
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttrs.append(len(set(window)) / window_size)
    
    return round(float(np.mean(ttrs)), 3)

def compute_comprehensive_metrics(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None

    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 0]

    N = len(words)
    S = len(sentences)

    if N < 3 or S == 0:
        return None

    V = len(set(words))

    # 1. Lexical Diversity Suite
    ttr = round(V / N, 3)
    guiraud_r = round(float(V / np.sqrt(N)), 2)
    herdan_c = round(float(math.log(V) / math.log(N)), 3) if N > 1 and V > 1 else 0.0
    maas_a = round(float((math.log(N) - math.log(V)) / (math.log(N) ** 2)), 3) if N > 1 and V > 1 else 0.0
    mtld = compute_mtld(words)
    mattr = compute_mattr(words)

    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    hapax_count = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = round((hapax_count / N) * 100, 1)

    function_count = sum(1 for w in words if w in FUNCTION_WORDS)
    content_count = N - function_count
    function_ratio = round((function_count / N) * 100, 1)
    content_ratio = round((content_count / N) * 100, 1)

    # 2. Syntactic Dependency & Corrected Passive Voice Analysis
    def get_node_depth(node):
        if not list(node.children):
            return 1
        return 1 + max(get_node_depth(child) for child in node.children)

    tree_depths = [get_node_depth(sent.root) for sent in doc.sents]
    mean_tree_depth = round(float(np.mean(tree_depths)), 2)
    max_tree_depth = int(np.max(tree_depths))

    dep_distances = []
    prepositional_phrases = 0
    total_verbs = 0
    passive_verbs = 0
    passive_sentences_count = 0

    for sent in doc.sents:
        sent_has_passive = False
        for token in sent:
            if token.head != token:
                dep_distances.append(abs(token.i - token.head.i))
            if token.pos_ in {"VERB", "AUX"}:
                total_verbs += 1
            if token.dep_ in {"passive", "auxpass", "nsubjpass"}:
                sent_has_passive = True
                passive_verbs += 1
            if token.pos_ == "ADP":
                prepositional_phrases += 1
        if sent_has_passive:
            passive_sentences_count += 1

    mean_dep_distance = round(float(np.mean(dep_distances)), 2) if dep_distances else 0.0
    
    # تصحيح دقيق لمعادلة المبني للمجهول: نسبة الجمل التي تحتوي مبني للمجهول (المقياس المعياري)
    passive_sentence_ratio = round((passive_sentences_count / S) * 100, 1)
    # نسبة أفعال المبني للمجهول من إجمالي الأفعال
    passive_verb_ratio = round((passive_verbs / max(1, total_verbs)) * 100, 1)

    prep_phrase_density = round((prepositional_phrases / N) * 100, 1)

    mls = round(N / S, 2)
    sent_lengths = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]
    sent_length_sd = round(float(np.std(sent_lengths)), 2) if len(sent_lengths) > 1 else 0.0

    # 3. POS Distribution, Entropy & Nominalization
    pos_counts = {}
    noun_count = 0
    verb_count = 0
    for token in doc:
        if token.is_alpha:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            if token.pos_ == "NOUN":
                noun_count += 1
            elif token.pos_ == "VERB":
                verb_count += 1

    pos_probs = [count / N for count in pos_counts.values()]
    pos_entropy = round(float(-sum(p * math.log2(p) for p in pos_probs if p > 0)), 3)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    pos_bigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}" for i in range(len(pos_tags)-1)]
    pos_transition_ratio = round(len(set(pos_bigrams)) / max(1, len(pos_bigrams)), 3)

    nominalization_ratio = round(noun_count / max(1, verb_count), 2)

    # 4. Punctuation Profile & Entropy
    punct_chars = {",": 0, ".": 0, ";": 0, ":": 0, "-": 0, "(": 0, ")": 0, "?": 0, "!": 0, '"': 0, "'": 0}
    total_punct = 0
    for char in text:
        if char in punct_chars:
            punct_chars[char] += 1
            total_punct += 1

    punct_density = round((total_punct / N) * 100, 1)
    
    punct_probs = [c / total_punct for c in punct_chars.values() if c > 0]
    punct_entropy = round(float(-sum(p * math.log2(p) for p in punct_probs)), 3) if punct_probs else 0.0

    # 5. Academic & Stylometric Vocabulary
    awl_count = sum(1 for w in words if w in AWL_KEYWORDS)
    awl_density = round((awl_count / N) * 100, 1)
    ai_buzzwords_count = sum(1 for w in words if w in AI_BUZZWORDS)

    # 6. Readability Suite
    syllables = sum(len(re.findall(r'[aeiouy]+', w)) for w in words)
    flesch_reading_ease = round(206.835 - 1.015 * (N / S) - 84.6 * (syllables / N), 1)
    flesch_kincaid_grade = round(0.39 * (N / S) + 11.8 * (syllables / N) - 15.59, 1)

    return {
        "words": N,
        "sentences": S,
        "ttr": ttr,
        "guiraud_r": guiraud_r,
        "herdan_c": herdan_c,
        "maas_a": maas_a,
        "mtld": mtld,
        "mattr": mattr,
        "hapax_ratio": hapax_ratio,
        "function_ratio": function_ratio,
        "content_ratio": content_ratio,
        "mls": mls,
        "sent_length_sd": sent_length_sd,
        "mean_tree_depth": mean_tree_depth,
        "max_tree_depth": max_tree_depth,
        "mean_dep_distance": mean_dep_distance,
        "passive_ratio": passive_sentence_ratio,  # النسب المؤكدة الصحيحة
        "passive_verb_ratio": passive_verb_ratio,
        "prep_phrase_density": prep_phrase_density,
        "pos_entropy": pos_entropy,
        "pos_transition_ratio": pos_transition_ratio,
        "nominalization_ratio": nominalization_ratio,
        "awl_density": awl_density,
        "ai_buzzwords_count": ai_buzzwords_count,
        "punct_density": punct_density,
        "punct_entropy": punct_entropy,
        "flesch_reading_ease": flesch_reading_ease,
        "readability_grade": max(1.0, min(20.0, flesch_kincaid_grade))
    }

def calculate_stylometric_vector_distance(m_a, m_b, m_c):
    if not (m_a and m_b and m_c):
        return 0.0

    keys = ['ttr', 'guiraud_r', 'mtld', 'mls', 'mean_tree_depth', 'passive_ratio', 'pos_entropy', 'awl_density']
    
    v_a = np.array([m_a[k] for k in keys])
    v_b = np.array([m_b[k] for k in keys])
    v_c = np.array([m_c[k] for k in keys])

    norm = np.linalg.norm(v_a) + 1e-9
    v_a, v_b, v_c = v_a / norm, v_b / norm, v_c / norm

    dist_c_a = np.linalg.norm(v_c - v_a)
    dist_b_a = np.linalg.norm(v_b - v_a)

    if (dist_c_a + dist_b_a) == 0:
        return 0.0

    signal_score = round(float((1.0 - (dist_c_a / (dist_c_a + dist_b_a))) * 100), 1)
    return max(0.0, min(100.0, signal_score))

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
            return 15.0, f"Status code: {res.status_code}"
        except Exception as e:
            return 15.0, str(e)

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Research Engine Live"}

@app.post("/analyze-single")
async def analyze_single_text(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    metrics = compute_comprehensive_metrics(text)
    if not metrics:
        raise HTTPException(status_code=400, detail="Text must contain valid words.")

    ai_score, err = await query_modal_editlens(text)

    predicted_class = "Pure AI" if ai_score >= 50.0 else ("AI-Humanized" if ai_score >= 20.0 else "Human Baseline")

    return {
        "metrics": metrics,
        "classification": {
            "predicted_class": predicted_class,
            "probabilities": {
                "human": round(max(0.0, 100.0 - ai_score), 1),
                "pure_ai": round(ai_score if predicted_class == "Pure AI" else ai_score * 0.5, 1),
                "ai_humanized": round(ai_score if predicted_class == "AI-Humanized" else max(0.0, 100.0 - abs(50.0 - ai_score) * 2), 1)
            },
            "sub_scores": {
                "lexical_authenticity": round(min(100.0, metrics["mtld"] / 1.2), 1),
                "syntactic_complexity": round(min(100.0, metrics["mean_tree_depth"] * 15), 1),
                "stylistic_entropy": round(min(100.0, metrics["pos_entropy"] * 25), 1)
            },
            "diagnostic_flags": [
                f"Model-Based Detection Score: {ai_score}% AI-associated stylistic footprint.",
                f"MTLD Richness Score: {metrics['mtld']}",
                f"Mean Dependency Tree Depth: {metrics['mean_tree_depth']}",
                f"Passive Sentence Ratio: {metrics['passive_ratio']}%"
            ]
        }
    }

@app.post("/analyze")
@app.post("/analyze-corpus")
async def analyze_corpora(request: Request):
    data = await request.json()

    human_text = data.get("human_text", "")
    ai_text = data.get("ai_text", "")
    humanized_text = data.get("humanized_text", "")

    inputs = [("human", human_text), ("pure_ai", ai_text), ("ai_humanized", humanized_text)]

    tasks = []
    keys = []
    texts = []

    for key, txt in inputs:
        if txt and txt.strip():
            keys.append(key)
            texts.append(txt)
            active_tasks = tasks.append(query_modal_editlens(txt))

    if not tasks:
        raise HTTPException(status_code=400, detail="Please provide text in at least one corpus.")

    ai_scores = await asyncio.gather(*tasks)

    metrics_result = {}
    for key, txt, (score, err) in zip(keys, texts, ai_scores):
        m = compute_comprehensive_metrics(txt)
        if m:
            metrics_result[key] = m

    residual_signal = calculate_stylometric_vector_distance(
        metrics_result.get("human"),
        metrics_result.get("pure_ai"),
        metrics_result.get("ai_humanized")
    )

    return {
        "status": "success",
        "residual_ai_footprint_percentage": residual_signal,
        "residual_ai_style_signal": residual_signal,
        "metrics": metrics_result
    }
