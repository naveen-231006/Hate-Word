"""
Tamil Offensive Language Detector — Interactive Web Demo
Uses fine-tuned MuRIL, XLM-RoBERTa, and mBERT with majority-voting ensemble.
"""
import os
import re
import json
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mode

# ============================================================
#  Configuration
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'models')
LABEL_NAMES = [
    'Not_offensive', 'Offensive_Untargeted', 'Offensive_Targeted_Individual',
    'Offensive_Targeted_Group', 'Offensive_Targeted_Other', 'not-Tamil'
]
MODEL_CONFIGS = {
    'mbert': {'name': 'mBERT', 'path': os.path.join(MODEL_DIR, 'mbert', 'best_model')},
    'muril': {'name': 'MuRIL', 'path': os.path.join(MODEL_DIR, 'muril', 'best_model')},
    'xlm-roberta': {'name': 'XLM-RoBERTa', 'path': os.path.join(MODEL_DIR, 'xlm-roberta', 'best_model')},
}

# ============================================================
#  Text Preprocessing (same as training pipeline)
# ============================================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
#  Load Models
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

models = {}
tokenizers = {}

for key, cfg in MODEL_CONFIGS.items():
    print(f"Loading {cfg['name']} from {cfg['path']}...")
    tokenizers[key] = AutoTokenizer.from_pretrained(cfg['path'])
    models[key] = AutoModelForSequenceClassification.from_pretrained(cfg['path'])
    models[key].to(device)
    models[key].eval()
    print(f"  {cfg['name']} loaded.")

print("All models loaded!")

# ============================================================
#  Prediction
# ============================================================
def predict(text):
    cleaned = preprocess_text(text)
    if not cleaned:
        return {'error': 'Empty text after preprocessing'}

    results = {}
    all_preds = []

    for key in MODEL_CONFIGS:
        tok = tokenizers[key]
        model = models[key]
        inputs = tok(cleaned, return_tensors='pt', truncation=True,
                     max_length=128, padding='max_length').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        pred_idx = int(np.argmax(probs))
        all_preds.append(pred_idx)

        results[key] = {
            'model': MODEL_CONFIGS[key]['name'],
            'prediction': LABEL_NAMES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        }

    # Ensemble
    ensemble_pred, _ = mode(all_preds, keepdims=False)
    ensemble_idx = int(ensemble_pred)
    avg_probs = np.mean([
        list(results[k]['probabilities'].values()) for k in results
    ], axis=0)
    results['ensemble'] = {
        'model': 'Ensemble (Majority Vote)',
        'prediction': LABEL_NAMES[ensemble_idx],
        'confidence': float(avg_probs[ensemble_idx]),
        'probabilities': {LABEL_NAMES[i]: float(avg_probs[i]) for i in range(len(LABEL_NAMES))}
    }

    # Agreement
    agreement = len(set(all_preds)) == 1

    return {
        'original_text': text,
        'cleaned_text': cleaned,
        'results': results,
        'agreement': agreement,
        'vote_count': sum(1 for p in all_preds if p == ensemble_idx)
    }

# ============================================================
#  Flask App
# ============================================================
app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tamil Offensive Language Detector</title>
    <meta name="description" content="Real-time Tamil and Tanglish offensive language detection using MuRIL, XLM-RoBERTa, and mBERT ensemble.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #06060b;
            --bg-card: rgba(14, 14, 22, 0.7);
            --bg-card-hover: rgba(20, 20, 32, 0.8);
            --border: rgba(55, 55, 75, 0.4);
            --border-hover: rgba(99, 102, 241, 0.4);
            --text-primary: #f0f0f5;
            --text-secondary: #8b8ba3;
            --text-muted: #52526a;
            --accent-indigo: #818cf8;
            --accent-violet: #a78bfa;
            --accent-rose: #fb7185;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --safe: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --danger-dark: #991b1b;
            --info: #6366f1;
            --glass: rgba(255, 255, 255, 0.03);
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --radius-xl: 28px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Animated mesh background */
        .bg-mesh {
            position: fixed;
            inset: 0;
            z-index: -1;
            background:
                radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,102,241,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 60% 80% at 90% 80%, rgba(168,85,247,0.05) 0%, transparent 60%),
                radial-gradient(ellipse 70% 50% at 50% 50%, rgba(16,185,129,0.03) 0%, transparent 60%);
        }

        .bg-grid {
            position: fixed;
            inset: 0;
            z-index: -1;
            background-image:
                linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
            background-size: 60px 60px;
            mask-image: radial-gradient(ellipse at center, black 30%, transparent 70%);
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 48px 24px 32px;
        }

        /* ---- HEADER ---- */
        .header {
            text-align: center;
            margin-bottom: 56px;
            position: relative;
        }

        .header-icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 20px;
            background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            border: 1px solid rgba(99,102,241,0.2);
            box-shadow: 0 8px 32px rgba(99,102,241,0.15);
        }

        .header h1 {
            font-size: 2.6rem;
            font-weight: 900;
            letter-spacing: -1.5px;
            line-height: 1.1;
            margin-bottom: 14px;
        }

        .header h1 .gradient {
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 40%, #fb7185 70%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            font-weight: 400;
            max-width: 600px;
            margin: 0 auto 20px;
            line-height: 1.6;
        }

        .tech-pills {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .pill {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 16px;
            border-radius: 100px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: var(--glass);
            color: var(--text-secondary);
            backdrop-filter: blur(8px);
        }

        .pill .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }

        .pill.indigo .dot { background: var(--accent-indigo); }
        .pill.violet .dot { background: var(--accent-violet); }
        .pill.rose .dot   { background: var(--accent-rose); }
        .pill.emerald .dot{ background: var(--accent-emerald); }

        /* ---- INPUT CARD ---- */
        .input-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            padding: 36px;
            backdrop-filter: blur(24px);
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }

        .input-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(168,85,247,0.3), transparent);
        }

        .input-card label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 14px;
        }

        .input-card label .icon { font-size: 16px; }

        textarea {
            width: 100%;
            min-height: 110px;
            background: rgba(6, 6, 11, 0.5);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 18px;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 1.02rem;
            line-height: 1.6;
            resize: vertical;
            transition: all 0.3s ease;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent-indigo);
            box-shadow: 0 0 0 4px rgba(99,102,241,0.1), 0 4px 20px rgba(99,102,241,0.08);
        }

        textarea::placeholder { color: var(--text-muted); }

        .controls-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 18px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 13px 30px;
            border-radius: var(--radius-md);
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            border: none;
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-analyze {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            color: white;
            box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.1);
            letter-spacing: 0.3px;
        }

        .btn-analyze:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(99,102,241,0.45), inset 0 1px 0 rgba(255,255,255,0.15);
        }

        .btn-analyze:active { transform: translateY(0); }

        .btn-analyze:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-clear {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid var(--border);
            padding: 12px 24px;
        }

        .btn-clear:hover {
            background: rgba(255,255,255,0.03);
            color: var(--text-secondary);
            border-color: rgba(255,255,255,0.1);
        }

        .shortcut-hint {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        kbd {
            padding: 2px 7px;
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-secondary);
        }

        /* Example chips */
        .examples-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .examples {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .chip {
            padding: 7px 16px;
            border-radius: 100px;
            font-size: 0.8rem;
            font-weight: 500;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .chip:hover {
            background: rgba(99,102,241,0.1);
            border-color: rgba(99,102,241,0.3);
            color: var(--accent-indigo);
            transform: translateY(-1px);
        }

        /* ---- LOADING ---- */
        #loading {
            display: none;
            text-align: center;
            padding: 60px 20px;
        }

        #loading.visible { display: block; }

        .loader {
            width: 48px;
            height: 48px;
            margin: 0 auto 20px;
            position: relative;
        }

        .loader::before, .loader::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            border: 3px solid transparent;
        }

        .loader::before {
            border-top-color: var(--accent-indigo);
            animation: spin 0.9s linear infinite;
        }

        .loader::after {
            border-bottom-color: var(--accent-violet);
            animation: spin 0.9s linear infinite reverse;
            inset: 6px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .loader-text {
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .loader-sub {
            color: var(--text-muted);
            font-size: 0.78rem;
            margin-top: 6px;
        }

        /* ---- RESULTS ---- */
        #results {
            display: none;
            animation: fadeUp 0.5s ease;
        }

        #results.visible { display: block; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* Ensemble Hero Card */
        .hero-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            padding: 40px 36px;
            backdrop-filter: blur(24px);
            margin-bottom: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .hero-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
        }

        .hero-card.safe::before    { background: linear-gradient(90deg, transparent, var(--safe), transparent); }
        .hero-card.warning::before { background: linear-gradient(90deg, transparent, var(--warning), transparent); }
        .hero-card.danger::before  { background: linear-gradient(90deg, transparent, var(--danger), transparent); }
        .hero-card.info::before    { background: linear-gradient(90deg, transparent, var(--info), transparent); }

        .hero-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 100px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 16px;
        }

        .hero-tag.safe    { background: rgba(16,185,129,0.12); color: var(--safe); border: 1px solid rgba(16,185,129,0.2); }
        .hero-tag.warning { background: rgba(245,158,11,0.12); color: var(--warning); border: 1px solid rgba(245,158,11,0.2); }
        .hero-tag.danger  { background: rgba(239,68,68,0.12); color: var(--danger); border: 1px solid rgba(239,68,68,0.2); }
        .hero-tag.info    { background: rgba(99,102,241,0.12); color: var(--info); border: 1px solid rgba(99,102,241,0.2); }

        .hero-prediction {
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: -0.5px;
            margin-bottom: 10px;
        }

        .hero-meta {
            display: flex;
            justify-content: center;
            gap: 24px;
            flex-wrap: wrap;
        }

        .hero-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .hero-stat .value {
            font-size: 1.3rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .hero-stat .label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 2px;
        }

        /* Vote indicator */
        .vote-dots {
            display: flex;
            gap: 6px;
            justify-content: center;
            margin-top: 16px;
        }

        .vote-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            border: 2px solid var(--border);
            transition: all 0.3s;
        }

        .vote-dot.agree { border-color: var(--safe); background: var(--safe); box-shadow: 0 0 8px rgba(16,185,129,0.4); }
        .vote-dot.disagree { border-color: var(--danger); background: var(--danger); }

        /* Model Cards Grid */
        .models-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 20px;
        }

        @media (max-width: 768px) {
            .models-grid { grid-template-columns: 1fr; }
        }

        .model-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .model-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-hover), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .model-card:hover {
            border-color: var(--border-hover);
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        }

        .model-card:hover::before { opacity: 1; }

        .model-card .mc-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .mc-name {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        .mc-badge {
            padding: 3px 10px;
            border-radius: 100px;
            font-size: 0.65rem;
            font-weight: 700;
        }

        .mc-pred {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .mc-conf {
            font-size: 0.82rem;
            color: var(--text-secondary);
            margin-bottom: 18px;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Probability bars */
        .prob-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .prob-name {
            font-size: 0.65rem;
            color: var(--text-muted);
            width: 90px;
            flex-shrink: 0;
            text-align: right;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .prob-track {
            flex: 1;
            height: 5px;
            background: rgba(255,255,255,0.04);
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .prob-pct {
            font-size: 0.65rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            width: 38px;
            text-align: right;
        }

        /* Preprocessed text box */
        .preprocess-box {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 18px 22px;
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }

        .preprocess-box .pp-icon {
            font-size: 16px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .preprocess-box .pp-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }

        .preprocess-box .pp-text {
            font-size: 0.88rem;
            color: var(--text-secondary);
            font-style: italic;
            font-family: 'JetBrains Mono', monospace;
            word-break: break-word;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 48px 0 32px;
            color: var(--text-muted);
            font-size: 0.78rem;
        }

        footer a {
            color: var(--accent-indigo);
            text-decoration: none;
            transition: color 0.2s;
        }

        footer a:hover { color: var(--accent-violet); }

        .footer-stats {
            display: flex;
            gap: 24px;
            justify-content: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .footer-stat {
            font-size: 0.72rem;
            color: var(--text-muted);
        }

        .footer-stat strong {
            color: var(--text-secondary);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="bg-mesh"></div>
    <div class="bg-grid"></div>

    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-icon">🛡️</div>
            <h1><span class="gradient">Tamil Offensive Language</span><br>Detector</h1>
            <p class="subtitle">Real-time offensive language detection for Tamil, Tanglish, and code-mixed text using an ensemble of three multilingual transformer models.</p>
            <div class="tech-pills">
                <span class="pill indigo"><span class="dot"></span>MuRIL · 236M</span>
                <span class="pill violet"><span class="dot"></span>XLM-RoBERTa · 278M</span>
                <span class="pill rose"><span class="dot"></span>mBERT · 177M</span>
                <span class="pill emerald"><span class="dot"></span>Ensemble F1: 0.762</span>
            </div>
        </div>

        <!-- Input -->
        <div class="input-card" id="inputCard">
            <label><span class="icon">✍️</span> Enter Tamil / Tanglish / Code-Mixed Text</label>
            <textarea id="inputText" placeholder="Type or paste text here…  e.g., 'Padam vera level mass iruku'"></textarea>
            <div class="controls-row">
                <button class="btn btn-analyze" id="analyzeBtn" onclick="analyze()">
                    ⚡ Analyze
                </button>
                <button class="btn btn-clear" onclick="clearAll()">Clear</button>
                <span class="shortcut-hint"><kbd>Enter</kbd> to analyze</span>
            </div>
            <div class="examples-label">Try these examples</div>
            <div class="examples">
                <span class="chip" onclick="setExample(this)">Padam vera level mass iruku</span>
                <span class="chip" onclick="setExample(this)">ithellam oru padama da</span>
                <span class="chip" onclick="setExample(this)">nalla iruku bro</span>
                <span class="chip" onclick="setExample(this)">ivan oru waste fellow</span>
                <span class="chip" onclick="setExample(this)">super acting in this movie</span>
                <span class="chip" onclick="setExample(this)">dei romba mokka da</span>
            </div>
        </div>

        <!-- Loading -->
        <div id="loading">
            <div class="loader"></div>
            <div class="loader-text">Analyzing with 3 models…</div>
            <div class="loader-sub">MuRIL · XLM-RoBERTa · mBERT</div>
        </div>

        <!-- Results -->
        <div id="results">
            <div class="hero-card" id="heroCard"></div>
            <div class="models-grid" id="modelsGrid"></div>
            <div class="preprocess-box" id="preprocessBox"></div>
        </div>

        <!-- Footer -->
        <footer>
            <div class="footer-stats">
                <span class="footer-stat">Dataset: <strong>35,138 samples</strong></span>
                <span class="footer-stat">Classes: <strong>6</strong></span>
                <span class="footer-stat">Models: <strong>3 + ensemble</strong></span>
            </div>
            Built for research · <a href="https://github.com/naveen-231006/Hate-Word" target="_blank">View on GitHub ↗</a>
        </footer>
    </div>

    <script>
        const COLORS = {
            'Not_offensive': '#10b981',
            'Offensive_Untargeted': '#f59e0b',
            'Offensive_Targeted_Individual': '#ef4444',
            'Offensive_Targeted_Group': '#dc2626',
            'Offensive_Targeted_Other': '#991b1b',
            'not-Tamil': '#6366f1'
        };

        const EMOJIS = {
            'Not_offensive': '✅',
            'Offensive_Untargeted': '⚠️',
            'Offensive_Targeted_Individual': '🎯',
            'Offensive_Targeted_Group': '👥',
            'Offensive_Targeted_Other': '❗',
            'not-Tamil': '🌐'
        };

        const SEVERITY = {
            'Not_offensive': 'safe',
            'Offensive_Untargeted': 'warning',
            'Offensive_Targeted_Individual': 'danger',
            'Offensive_Targeted_Group': 'danger',
            'Offensive_Targeted_Other': 'danger',
            'not-Tamil': 'info'
        };

        const SHORT = {
            'Not_offensive': 'Not Offensive',
            'Offensive_Untargeted': 'Offensive (Untargeted)',
            'Offensive_Targeted_Individual': 'Targeted: Individual',
            'Offensive_Targeted_Group': 'Targeted: Group',
            'Offensive_Targeted_Other': 'Targeted: Other',
            'not-Tamil': 'Not Tamil'
        };

        function setExample(el) {
            document.getElementById('inputText').value = el.textContent;
            document.getElementById('inputText').focus();
        }

        function clearAll() {
            document.getElementById('inputText').value = '';
            document.getElementById('results').classList.remove('visible');
            document.getElementById('inputText').focus();
        }

        async function analyze() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) return;

            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Analyzing…';
            document.getElementById('loading').classList.add('visible');
            document.getElementById('results').classList.remove('visible');

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await resp.json();
                renderResults(data);
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = '⚡ Analyze';
                document.getElementById('loading').classList.remove('visible');
            }
        }

        function renderResults(data) {
            const {results, cleaned_text, agreement, vote_count} = data;
            const ens = results.ensemble;
            const sev = SEVERITY[ens.prediction];
            const color = COLORS[ens.prediction];

            // Hero card
            const heroCard = document.getElementById('heroCard');
            heroCard.className = `hero-card ${sev}`;
            heroCard.innerHTML = `
                <span class="hero-tag ${sev}">${EMOJIS[ens.prediction]} Ensemble Prediction</span>
                <div class="hero-prediction" style="color:${color}">${SHORT[ens.prediction]}</div>
                <div class="hero-meta">
                    <div class="hero-stat">
                        <span class="value" style="color:${color}">${(ens.confidence*100).toFixed(1)}%</span>
                        <span class="label">Avg Confidence</span>
                    </div>
                    <div class="hero-stat">
                        <span class="value">${vote_count}/3</span>
                        <span class="label">Models Agree</span>
                    </div>
                </div>
                <div class="vote-dots">
                    ${['mbert','muril','xlm-roberta'].map(k => {
                        const agrees = results[k].prediction === ens.prediction;
                        return `<span class="vote-dot ${agrees?'agree':'disagree'}" title="${results[k].model}: ${results[k].prediction}"></span>`;
                    }).join('')}
                </div>
            `;

            // Model cards
            const grid = document.getElementById('modelsGrid');
            grid.innerHTML = '';
            for (const key of ['mbert','muril','xlm-roberta']) {
                const r = results[key];
                const c = COLORS[r.prediction];
                const s = SEVERITY[r.prediction];
                const agrees = r.prediction === ens.prediction;

                let probsHTML = '';
                const sortedProbs = Object.entries(r.probabilities).sort((a,b) => b[1]-a[1]);
                for (const [label, prob] of sortedProbs) {
                    const pct = (prob*100).toFixed(1);
                    probsHTML += `
                        <div class="prob-row">
                            <span class="prob-name">${SHORT[label].split(':').pop().trim()}</span>
                            <div class="prob-track">
                                <div class="prob-fill" style="width:${pct}%;background:${COLORS[label]}"></div>
                            </div>
                            <span class="prob-pct">${pct}%</span>
                        </div>`;
                }

                grid.innerHTML += `
                    <div class="model-card">
                        <div class="mc-header">
                            <span class="mc-name">${r.model}</span>
                            <span class="mc-badge" style="background:${agrees?'rgba(16,185,129,0.12)':'rgba(239,68,68,0.12)'};color:${agrees?'#10b981':'#ef4444'}">${agrees?'✓ Agree':'✗ Differ'}</span>
                        </div>
                        <div class="mc-pred" style="color:${c}">${EMOJIS[r.prediction]} ${SHORT[r.prediction]}</div>
                        <div class="mc-conf">${(r.confidence*100).toFixed(1)}% confidence</div>
                        ${probsHTML}
                    </div>`;
            }

            // Preprocessed text
            document.getElementById('preprocessBox').innerHTML = `
                <span class="pp-icon">🔧</span>
                <div>
                    <div class="pp-label">Preprocessed Input</div>
                    <div class="pp-text">${cleaned_text}</div>
                </div>`;

            document.getElementById('results').classList.add('visible');
        }

        document.getElementById('inputText').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyze();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = predict(text)
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Tamil Offensive Language Detector")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
