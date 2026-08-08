from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def dummy_payload(score, label):
    return {
        "metrics": {"words": 100, "sentences": 5, "guiraud_r": 5.0, "mls": 20.0, "pos_transition_ratio": 0.8},
        "classification": {
            "predicted_class": label,
            "confidence": "high",
            "probabilities": {"human": 100 - score, "pure_ai": score, "ai_humanized": 0},
            "sub_scores": {"lexical_authenticity": 70, "syntactic_complexity": 30, "stylistic_entropy": 60},
            "diagnostic_flags": [f"Test Mode: {score}% AI score"],
            "disclaimer": "Test Mode Active"
        }
    }

@app.get("/")
def home():
    return {"status": "Active"}

@app.post("/analyze")
@app.post("/analyze-corpus")
@app.post("/analyze-single")
async def analyze_mock(request: Request):
    res = {
        "corpus_a": dummy_payload(15.0, "Human Baseline"),
        "corpus_b": dummy_payload(95.0, "Pure AI"),
        "corpus_c": dummy_payload(50.0, "AI-Humanized")
    }
    return {"status": "success", "results": res, **res}
