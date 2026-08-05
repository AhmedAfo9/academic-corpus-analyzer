from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import numpy as np
import re

app = FastAPI(title="Academic Corpus Analyzer API")

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

class CorpusInput(BaseModel):
    human_text: str
    ai_text: str
    humanized_text: str

def analyze_single_corpus(text: str):
    if not text.strip():
        return {
            "words": 0, "sentences": 0, "ttr": 0, "mls": 0, 
            "lexical_density": 0, "passive_ratio": 0, 
            "burstiness": 0, "ai_words_count": 0
        }

    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
    
    total_words = len(words)
    total_sentences = len(sentences)
    
    if total_words == 0 or total_sentences == 0:
        return {
            "words": 0, "sentences": 0, "ttr": 0, "mls": 0, 
            "lexical_density": 0, "passive_ratio": 0, 
            "burstiness": 0, "ai_words_count": 0
        }

    ttr = round(len(set(words)) / total_words, 3)
    sentence_lengths = [len([t for t in nlp(s) if t.is_alpha]) for s in sentences]
    mls = round(total_words / total_sentences, 2)
    burstiness = round(float(np.std(sentence_lengths)), 2) if len(sentence_lengths) > 1 else 0.0

    content_words = [token for token in doc if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV")]
    lexical_density = round((len(content_words) / total_words) * 100, 2)

    passive_instances = sum(1 for token in doc if token.dep_ in ("nsubjpass", "auxpass"))
    passive_ratio = round((passive_instances / total_sentences) * 100, 2)

    text_lower = text.lower()
    ai_words_count = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower)) for word in AI_BUZZWORDS)

    return {
        "words": total_words,
        "sentences": total_sentences,
        "ttr": ttr,
        "mls": mls,
        "lexical_density": lexical_density,
        "passive_ratio": passive_ratio,
        "burstiness": burstiness,
        "ai_words_count": ai_words_count
    }

@app.get("/")
def home():
    return {"status": "Academic Corpus Analyzer API is Running"}

@app.post("/analyze")
def analyze_corpora(data: CorpusInput):
    human_res = analyze_single_corpus(data.human_text)
    ai_res = analyze_single_corpus(data.ai_text)
    humanized_res = analyze_single_corpus(data.humanized_text)

    residual_footprint = 100.0
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
