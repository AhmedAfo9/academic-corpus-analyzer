import os
import re
import io
import math
import asyncio
import numpy as np
import spacy
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pypdf
import docx

app = FastAPI(
    title="Academic Corpus Analyzer - Hybrid Transformer Engine",
    description="Multidimensional Linguistic, Stylometric & Document Forensics Platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODAL_EDITLENS_URL = "https://ahmedfalahoraibi--editlens-engine-editlensserver-predict.modal.run"
MODAL_SEMANTIC_URL = "https://ahmedfalahoraibi--academic-semantic-engine-calc-semantic-sim.modal.run"

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

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                extracted_pages.append(t)
        return "\n".join(extracted_pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF document: {str(e)}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse DOCX document: {str(e)}")

def split_text_into_chunks(text: str, target_chunk_words: int = 500) -> List[Dict[str, Any]]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_words = []
    current_count = 0
    chunk_index = 1

    for p in paragraphs:
        p_words = p.split()
        if not p_words:
            continue
        current_words.extend(p_words)
        current_count += len(p_words)

        if current_count >= target_chunk_words:
            chunk_str = " ".join(current_words)
            chunks.append({
                "chunk_id": chunk_index,
                "word_count": len(current_words),
                "text": chunk_str
            })
            chunk_index += 1
            current_words = []
            current_count = 0

    if current_words:
        chunk_str = " ".join(current_words)
        if chunks and len(current_words) < 150:
            chunks[-1]["text"] += " " + chunk_str
            chunks[-1]["word_count"] += len(current_words)
        else:
            chunks.append({
                "chunk_id": chunk_index,
                "word_count": len(current_words),
                "text": chunk_str
            })

    return chunks

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

async def query_modal_semantic(text_a: str, text_b: str, text_c: str) -> Dict[str, float]:
    if not (text_a and text_b and text_c):
        return {"sim_a_b": 0.0, "sim_a_c": 0.0, "sim_b_c": 0.0}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            res = await client.post(MODAL_SEMANTIC_URL, json={"text_a": text_a, "text_b": text_b, "text_c": text_c})
            if res.status_code == 200:
                return res.json()
            return {"sim_a_b": 0.0, "sim_a_c": 0.0, "sim_b_c": 0.0}
        except Exception:
            return {"sim_a_b": 0.0, "sim_a_c": 0.0, "sim_b_c": 0.0}

def compute_burrows_delta(text_a: str, text_b: str, text_c: str) -> Dict[str, float]:
    if not (text_a and text_b and text_c):
        return {"delta_a_b": 0.0, "delta_a_c": 0.0, "delta_b_c": 0.0}
    
    def get_fw_freqs(txt):
        tokens = [t.text.lower() for t in nlp(txt) if t.is_alpha]
        tot = max(1, len(tokens))
        return {fw: tokens.count(fw) / tot for fw in FUNCTION_WORDS}

    f_a = get_fw_freqs(text_a)
    f_b = get_fw_freqs(text_b)
    f_c = get_fw_freqs(text_c)

    fw_list = list(FUNCTION_WORDS)
    mat = np.array([
        [f_a[fw] for fw in fw_list],
        [f_b[fw] for fw in fw_list],
        [f_c[fw] for fw in fw_list]
    ])
    
    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0) + 1e-9
    z_mat = (mat - means) / stds

    return {
        "delta_a_b": round(float(np.mean(np.abs(z_mat[0] - z_mat[1]))), 3),
        "delta_a_c": round(float(np.mean(np.abs(z_mat[0] - z_mat[2]))), 3),
        "delta_b_c": round(float(np.mean(np.abs(z_mat[1] - z_mat[2]))), 3)
    }

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

    def get_node_depth(node):
        if not list(node.children):
            return 1
        return 1 + max(get_node_depth(child) for child in node.children)

    tree_depths = [get_node_depth(sent.root) for sent in doc.sents]
    mean_tree_depth = round(float(np.mean(tree_depths)), 2)

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
    passive_sentence_ratio = round((passive_sentences_count / S) * 100, 1)
    passive_verb_ratio = round((passive_verbs / max(1, total_verbs)) * 100, 1)

    prep_phrase_density = round((prepositional_phrases / N) * 100, 1)

    mls = round(N / S, 2)
    sent_lengths = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]
    sent_length_sd = round(float(np.std(sent_lengths)), 2) if len(sent_lengths) > 1 else 0.0

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

    punct_chars = {",": 0, ".": 0, ";": 0, ":": 0, "-": 0, "(": 0, ")": 0, "?": 0, "!": 0, '"': 0, "'": 0}
    total_punct = 0
    for char in text:
        if char in punct_chars:
            punct_chars[char] += 1
            total_punct += 1

    punct_density = round((total_punct / N) * 100, 1)
    punct_probs = [c / total_punct for c in punct_chars.values() if c > 0]
    punct_entropy = round(float(-sum(p * math.log2(p) for p in punct_probs)), 3) if punct_probs else 0.0

    awl_count = sum(1 for w in words if w in AWL_KEYWORDS)
    awl_density = round((awl_count / N) * 100, 1)
    ai_buzzwords_count = sum(1 for w in words if w in AI_BUZZWORDS)

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
        "mean_dep_distance": mean_dep_distance,
        "passive_ratio": passive_sentence_ratio,
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

async def process_single_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    c_text = chunk["text"]
    metrics = compute_comprehensive_metrics(c_text)
    ai_score, _ = await query_modal_editlens(c_text)

    if not metrics:
        return {
            "chunk_id": chunk["chunk_id"],
            "word_count": chunk["word_count"],
            "ai_score": ai_score,
            "classification": "Human Baseline",
            "metrics": {}
        }

    is_lexically_degraded = (metrics["mtld"] < 65.0) or (metrics["ttr"] < 0.52)
    is_punct_stripped = metrics["punct_density"] < 14.0
    is_shortened = metrics["mls"] < 16.0

    if ai_score >= 45.0:
        if is_lexically_degraded or is_punct_stripped:
            predicted_class = "AI-Humanized"
        else:
            predicted_class = "Pure AI"
    elif ai_score >= 20.0:
        predicted_class = "AI-Humanized"
    else:
        if is_lexically_degraded and is_shortened:
            predicted_class = "AI-Humanized"
        else:
            predicted_class = "Human Baseline"

    return {
        "chunk_id": chunk["chunk_id"],
        "word_count": chunk["word_count"],
        "ai_score": ai_score,
        "classification": predicted_class,
        "metrics": metrics
    }

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer - Hybrid Transformer Engine Active"}

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

    is_lexically_degraded = (metrics["mtld"] < 65.0) or (metrics["ttr"] < 0.52)
    is_punct_stripped = metrics["punct_density"] < 14.0
    is_shortened_sentences = metrics["mls"] < 16.0

    diagnostic_flags = [
        f"Neural Model Prediction: {ai_score}% raw AI probability footprint."
    ]

    if ai_score >= 45.0:
        if is_lexically_degraded or is_punct_stripped:
            predicted_class = "AI-Humanized"
            diagnostic_flags.append("Humanizer Anomaly Detected: High neural footprint overridden by lexical degradation (MTLD/TTR drop) & punctuation stripping.")
        else:
            predicted_class = "Pure AI"
            diagnostic_flags.append("Pure AI Signature: High neural trace combined with elevated MTLD lexical richness.")
    elif ai_score >= 20.0:
        predicted_class = "AI-Humanized"
        diagnostic_flags.append("Moderate Neural Trace: Text exhibits stylistic modifications characteristic of paraphrasing engines.")
    else:
        if is_lexically_degraded and is_shortened_sentences:
            predicted_class = "AI-Humanized"
            diagnostic_flags.append("Low Neural Trace but High Stylometric Anomaly: Sentence structure and vocabulary degradation indicate deep automated paraphrasing.")
        else:
            predicted_class = "Human Baseline"
            diagnostic_flags.append("Human Baseline: Low neural footprint with natural lexical richness and syntactic variance.")

    if predicted_class == "AI-Humanized":
        prob_humanized = round(max(55.0, ai_score), 1)
        prob_pure_ai = round(min(40.0, ai_score * 0.4), 1)
        prob_human = round(max(0.0, 100.0 - prob_humanized - prob_pure_ai), 1)
    elif predicted_class == "Pure AI":
        prob_pure_ai = round(max(60.0, ai_score), 1)
        prob_humanized = round((100.0 - prob_pure_ai) * 0.7, 1)
        prob_human = round(max(0.0, 100.0 - prob_pure_ai - prob_humanized), 1)
    else:
        prob_human = round(max(75.0, 100.0 - ai_score), 1)
        prob_humanized = round((100.0 - prob_human) * 0.6, 1)
        prob_pure_ai = round(max(0.0, 100.0 - prob_human - prob_humanized), 1)

    return {
        "metrics": metrics,
        "classification": {
            "predicted_class": predicted_class,
            "probabilities": {
                "human": prob_human,
                "pure_ai": prob_pure_ai,
                "ai_humanized": prob_humanized
            },
            "sub_scores": {
                "lexical_authenticity": round(min(100.0, metrics["mtld"] / 1.2), 1),
                "syntactic_complexity": round(min(100.0, metrics["mean_tree_depth"] * 15), 1),
                "stylistic_entropy": round(min(100.0, metrics["pos_entropy"] * 25), 1)
            },
            "diagnostic_flags": diagnostic_flags
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
            tasks.append(query_modal_editlens(txt))

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

    semantic_sim = await query_modal_semantic(human_text, ai_text, humanized_text)
    burrows_delta = compute_burrows_delta(human_text, ai_text, humanized_text)

    return {
        "status": "success",
        "residual_ai_footprint_percentage": residual_signal,
        "residual_ai_style_signal": residual_signal,
        "semantic_similarity": semantic_sim,
        "burrows_delta": burrows_delta,
        "metrics": metrics_result
    }

@app.post("/analyze-document")
async def analyze_uploaded_document(file: UploadFile = File(...)):
    max_bytes = 30 * 1024 * 1024  # 30 MB
    content = await file.read()
    
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail="File size exceeds the 30MB limit.")

    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(content)
    elif filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Only PDF and DOCX files are allowed.")

    if not extracted_text or len(extracted_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document appears to be empty or contains no readable text.")

    chunks = split_text_into_chunks(extracted_text, target_chunk_words=500)
    
    # Process chunks in parallel using asyncio.gather
    chunk_tasks = [process_single_chunk(c) for c in chunks]
    chunk_results = await asyncio.gather(*chunk_tasks)

    total_words = sum(c["word_count"] for c in chunk_results)
    estimated_pages = max(1, math.ceil(total_words / 350))

    # Cross-Chunk Longitudinal Forensic Metrics
    ai_scores = [c["ai_score"] for c in chunk_results]
    avg_ai_score = round(float(np.mean(ai_scores)), 1) if ai_scores else 0.0

    pos_entropies = [c["metrics"]["pos_entropy"] for c in chunk_results if c.get("metrics") and "pos_entropy" in c["metrics"]]
    tree_depths = [c["metrics"]["mean_tree_depth"] for c in chunk_results if c.get("metrics") and "mean_tree_depth" in c["metrics"]]
    mls_values = [c["metrics"]["mls"] for c in chunk_results if c.get("metrics") and "mls" in c["metrics"]]

    std_pos_entropy = round(float(np.std(pos_entropies)), 3) if len(pos_entropies) > 1 else 0.15
    std_tree_depth = round(float(np.std(tree_depths)), 2) if len(tree_depths) > 1 else 0.8
    std_mls = round(float(np.std(mls_values)), 2) if len(mls_values) > 1 else 4.0

    # Detect Grounded AI (Ultra-low variance + persistent artificial structure)
    # Human text naturally fluctuates (std_pos > 0.12, std_tree > 0.6).
    # AI Grounded text stays unnaturally consistent (std_pos < 0.07, std_tree < 0.4).
    is_unnaturally_consistent = (std_pos_entropy < 0.08) and (std_tree_depth < 0.45)
    
    if is_unnaturally_consistent and len(chunks) >= 3:
        grounded_ai_risk = round(max(65.0, 100.0 - (std_pos_entropy * 500) - (std_tree_depth * 40)), 1)
        grounded_ai_flag = "Grounded AI Detected: Ultra-low cross-chunk variance reveals source-constrained artificial generation across pages."
    else:
        grounded_ai_risk = round(min(35.0, std_pos_entropy * 100), 1)
        grounded_ai_flag = "Natural Human Variance: Document exhibits organic stylistic fluctuation across sections."

    # Breakdown Counts for Heatmap
    human_count = sum(1 for c in chunk_results if c["classification"] == "Human Baseline")
    humanized_count = sum(1 for c in chunk_results if c["classification"] == "AI-Humanized")
    pure_ai_count = sum(1 for c in chunk_results if c["classification"] == "Pure AI")

    overall_class = "Human Baseline"
    if grounded_ai_risk >= 60.0 or pure_ai_count >= len(chunks) * 0.4:
        overall_class = "Pure AI / Grounded AI"
    elif humanized_count >= len(chunks) * 0.35:
        overall_class = "AI-Humanized"

    return {
        "status": "success",
        "filename": file.filename,
        "file_size_mb": round(len(content) / (1024 * 1024), 2),
        "total_words": total_words,
        "estimated_pages": estimated_pages,
        "total_chunks": len(chunks),
        "overall_classification": overall_class,
        "overall_ai_footprint_avg": avg_ai_score,
        "grounded_ai_risk_score": grounded_ai_risk,
        "grounded_ai_flag": grounded_ai_flag,
        "longitudinal_variance": {
            "std_pos_entropy": std_pos_entropy,
            "std_tree_depth": std_tree_depth,
            "std_mls": std_mls
        },
        "heatmap_distribution": {
            "human_chunks": human_count,
            "humanized_chunks": humanized_count,
            "pure_ai_chunks": pure_ai_count
        },
        "chunk_breakdown": [
            {
                "chunk_id": c["chunk_id"],
                "word_count": c["word_count"],
                "ai_score": c["ai_score"],
                "classification": c["classification"],
                "page_estimate": math.ceil(c["chunk_id"] * 500 / 350)
            }
            for c in chunk_results
        ]
    }
