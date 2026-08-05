from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import numpy as np
import re
import textstat
from collections import Counter

app = FastAPI(title="Academic Corpus Analyzer - Ultimate Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

AI_BUZZWORDS = [
    "delve", "tapestry", "multifaceted", "pivotal", "crucial", "paramount",
    "testament", "beacon", "underscores", "interplay", "fostering", "vibrant",
    "comprehensive", "in conclusion", "it is worth noting", "furthermore"
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
    def walk_tree(node):
        if not list(node.children):
            return 1
        return 1 + max(walk_tree(child) for child in node.children)
    roots = [token for token in sent_doc if token.head == token]
    return max([walk_tree(root) for root in roots]) if roots else 1

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

    # Vocabulary Metrics
    unique_words = len(set(words))
    ttr = round(unique_words / total_words, 3)
    guiraud_r = round(unique_words / np.sqrt(total_words), 2) # Length-independent richness
    
    word_counts = Counter(words)
    hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
    hapax_ratio = round((hapax_count / total_words) * 100, 2)

    sentence_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    mls = round(total_words / total_sentences, 2)
    burstiness = round(float(np.std(sentence_lengths)), 2) if len(sentence_lengths) > 1 else 0.0

    content_words = [token for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lexical_density = round((len(content_words) / total_words) * 100, 2)

    passive_instances = sum(1 for token in doc if token.dep_ in ("nsubjpass", "auxpass"))
    passive_ratio = min(round((passive_instances / total_sentences) * 100, 2), 100.0)

    text_lower = text.lower()
    ai_words_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in AI_BUZZWORDS)

    awl_count = sum(1 for w in words if w in AWL_WORDS)
    awl_density = round((awl_count / total_words) * 100, 2)

    tree_depths = [get_sentence_depth(s) for s in sentences]
    avg_tree_depth = round(sum(tree_depths) / len(tree_depths), 2)

    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    pos_bigrams = [f"{pos_tags[i]}_{pos_tags[i+1]}" for i in range(len(pos_tags)-1)]
    pos_transition_ratio = round(len(set(pos_bigrams)) / len(pos_bigrams), 3) if pos_bigrams else 0.0

    readability_grade = round(textstat.flesch_kincaid_grade(text), 2)

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
        "readability_grade": readability_grade
    }

def classify_single_text_logic(m):
    # Advanced Multi-Dimensional Distance Scoring
    w_human = 0.0
    w_ai = 0.0
    w_humanized = 0.0
    flags = []

    # 1. Lexical Sub-Score Evaluation
    if m["lexical_density"] < 43.0:
        w_humanized += 30.0
        flags.append("Degraded Lexical Density (<43%): Strong indicator of functional padding used by AI Humanizers.")
    elif m["lexical_density"] > 48.0:
        w_human += 25.0
        flags.append("High Content-Word Density (>48%): Authentic human informative packaging.")

    if m["hapax_ratio"] < 35.0:
        w_humanized += 20.0
        w_ai += 15.0
        flags.append("Low Hapax Legomena Ratio (<35%): Repetitive core vocabulary usage.")
    elif m["hapax_ratio"] >= 42.0:
        w_human += 20.0
        flags.append("High Unique Vocabulary Spread (Hapax >= 42%): Natural lexical spontaneity.")

    # 2. Syntactic & Academic Register
    if m["awl_density"] > 1.4 and m["ttr"] < 0.52:
        w_humanized += 30.0
        flags.append("AWL Synonym Inflation with Low TTR: Disproportionate formal word swapping detected.")
    elif m["awl_density"] > 2.2 and m["mls"] > 18.0:
        w_ai += 35.0
        flags.append("Elevated AWL Density with Extended MLS (>18): Standard Pure AI signature.")

    # 3. Structural Variation & Entropy
    if m["pos_transition_ratio"] < 0.33:
        w_ai += 20.0
        w_humanized += 20.0
        flags.append("Low POS Transition Entropy (<0.33): Rigid, predictable grammatical transitions.")
    elif m["pos_transition_ratio"] >= 0.35:
        w_human += 25.0
        flags.append("High Structural POS Entropy (>=0.35): Varied human sentence architecture.")

    if m["ai_words_count"] > 0:
        w_ai += 40.0
        flags.append(f"Overt AI Transitional Buzzwords ({m['ai_words_count']} detected).")

    if m["readability_grade"] < 8.5 and m["awl_density"] > 1.2:
        w_humanized += 20.0
        flags.append("Readability Downgrade with Academic Terms: Structural simplification artifact.")

    # Calculate Probability Percentages
    base_score = 33.3
    tot_human = max(0.0, base_score + w_human)
    tot_ai = max(0.0, base_score + w_ai)
    tot_humanized = max(0.0, base_score + w_humanized)
    
    total = tot_human + tot_ai + tot_humanized
    prob_human = round((tot_human / total) * 100, 1)
    prob_ai = round((tot_ai / total) * 100, 1)
    prob_humanized = round((tot_humanized / total) * 100, 1)

    probs = {"Human Baseline": prob_human, "Pure AI": prob_ai, "AI-Humanized": prob_humanized}
    predicted_class = max(probs, key=probs.get)

    # Sub-scores calculation for deep analytics
    lexical_score = round(min(100.0, max(0.0, (m["guiraud_r"] * 10) + (m["lexical_density"] * 0.8))), 1)
    syntactic_score = round(min(100.0, max(0.0, (m["avg_tree_depth"] * 12) + (m["mls"] * 1.5))), 1)
    entropy_score = round(min(100.0, max(0.0, m["pos_transition_ratio"] * 250)), 1)

    return {
        "predicted_class": predicted_class,
        "probabilities": {
            "human": prob_human,
            "pure_ai": prob_ai,
            "ai_humanized": prob_humanized
        },
        "sub_scores": {
            "lexical_authenticity": lexical_score,
            "syntactic_complexity": syntactic_score,
            "stylistic_entropy": entropy_score
        },
        "diagnostic_flags": flags
    }

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer Ultimate API is Live"}

@app.post("/analyze")
def analyze_corpora(data: CorpusInput):
    human_res = analyze_single_corpus(data.human_text)
    ai_res = analyze_single_corpus(data.ai_text)
    humanized_res = analyze_single_corpus(data.humanized_text)

    valid_count = sum(1 for r in [human_res, ai_res, humanized_res] if r is not None)
    if valid_count < 2:
        raise HTTPException(status_code=400, detail="At least 2 non-empty corpora are required for comparative analysis.")

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
            "ai_humanized": humanized_res
        },
        "residual_ai_footprint_percentage": residual_footprint
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
        "classification": classification
    }
