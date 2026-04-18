"""
Tamil Offensive Language Detector — Interactive Web Demo
Uses fine-tuned MuRIL, XLM-RoBERTa, and mBERT with majority-voting ensemble.
"""
import os, re, json, torch, numpy as np
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import mode

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

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
models_loaded = {}
tokenizers_loaded = {}
for key, cfg in MODEL_CONFIGS.items():
    print(f"Loading {cfg['name']}...")
    tokenizers_loaded[key] = AutoTokenizer.from_pretrained(cfg['path'])
    models_loaded[key] = AutoModelForSequenceClassification.from_pretrained(cfg['path'])
    models_loaded[key].to(device).eval()
    print(f"  {cfg['name']} loaded.")
print("All models loaded!")

def predict(text):
    cleaned = preprocess_text(text)
    if not cleaned: return {'error': 'Empty text after preprocessing'}
    results = {}
    all_preds = []
    for key in MODEL_CONFIGS:
        inputs = tokenizers_loaded[key](cleaned, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
        with torch.no_grad():
            logits = models_loaded[key](**inputs).logits[0].cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        pred_idx = int(np.argmax(probs))
        all_preds.append(pred_idx)
        results[key] = {
            'model': MODEL_CONFIGS[key]['name'], 'prediction': LABEL_NAMES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        }
    ens_pred, _ = mode(all_preds, keepdims=False)
    ens_idx = int(ens_pred)
    avg_probs = np.mean([list(results[k]['probabilities'].values()) for k in results], axis=0)
    results['ensemble'] = {
        'model': 'Ensemble', 'prediction': LABEL_NAMES[ens_idx],
        'confidence': float(avg_probs[ens_idx]),
        'probabilities': {LABEL_NAMES[i]: float(avg_probs[i]) for i in range(len(LABEL_NAMES))}
    }
    return {'original_text': text, 'cleaned_text': cleaned, 'results': results,
            'agreement': len(set(all_preds)) == 1,
            'vote_count': sum(1 for p in all_preds if p == ens_idx)}

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tamil Offensive Language Detector</title>
<meta name="description" content="Real-time Tamil offensive language detection with ensemble transformers.">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ====== Theme Variables ====== */
:root, [data-theme="light"] {
    --bg: #f8f9fc;
    --bg-alt: #ffffff;
    --bg-card: #ffffff;
    --bg-input: #f1f3f8;
    --bg-hover: #f5f6fa;
    --border: #e2e5ef;
    --border-hover: #c7cad9;
    --text: #1a1d2e;
    --text-secondary: #5a5f7a;
    --text-muted: #8e93ac;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 12px 48px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    --shadow-xl: 0 20px 60px rgba(0,0,0,0.1);
    --accent: #6366f1;
    --accent-light: rgba(99,102,241,0.08);
    --accent-border: rgba(99,102,241,0.2);
    --accent-hover: #4f46e5;
    --gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    --dot-bg: rgba(99,102,241,0.06);
    --mesh-1: rgba(99,102,241,0.04);
    --mesh-2: rgba(168,85,247,0.03);
    --mesh-3: rgba(236,72,153,0.02);
    --toggle-bg: #e2e5ef;
    --toggle-dot: #ffffff;
    --safe: #059669; --safe-bg: #ecfdf5; --safe-border: #a7f3d0;
    --warn: #d97706; --warn-bg: #fffbeb; --warn-border: #fde68a;
    --danger: #dc2626; --danger-bg: #fef2f2; --danger-border: #fecaca;
    --info: #4f46e5; --info-bg: #eef2ff; --info-border: #c7d2fe;
}

[data-theme="dark"] {
    --bg: #0a0a12;
    --bg-alt: #0f0f1a;
    --bg-card: rgba(16,16,28,0.8);
    --bg-input: rgba(8,8,16,0.6);
    --bg-hover: rgba(22,22,36,0.8);
    --border: rgba(50,50,72,0.5);
    --border-hover: rgba(99,102,241,0.35);
    --text: #eeeef5;
    --text-secondary: #9295ad;
    --text-muted: #5a5d76;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
    --shadow-lg: 0 12px 48px rgba(0,0,0,0.5);
    --shadow-xl: 0 20px 60px rgba(0,0,0,0.6);
    --accent: #818cf8;
    --accent-light: rgba(99,102,241,0.12);
    --accent-border: rgba(99,102,241,0.25);
    --accent-hover: #a5b4fc;
    --gradient: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    --dot-bg: rgba(99,102,241,0.08);
    --mesh-1: rgba(99,102,241,0.06);
    --mesh-2: rgba(168,85,247,0.04);
    --mesh-3: rgba(236,72,153,0.03);
    --toggle-bg: #2a2a3e;
    --toggle-dot: #818cf8;
    --safe: #34d399; --safe-bg: rgba(16,185,129,0.1); --safe-border: rgba(16,185,129,0.25);
    --warn: #fbbf24; --warn-bg: rgba(245,158,11,0.1); --warn-border: rgba(245,158,11,0.25);
    --danger: #f87171; --danger-bg: rgba(239,68,68,0.1); --danger-border: rgba(239,68,68,0.25);
    --info: #818cf8; --info-bg: rgba(99,102,241,0.1); --info-border: rgba(99,102,241,0.25);
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    transition: background 0.4s ease, color 0.4s ease;
    -webkit-font-smoothing: antialiased;
}

/* Background decorations */
.bg-decor {
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
        radial-gradient(ellipse 80% 60% at 15% 25%, var(--mesh-1) 0%, transparent 55%),
        radial-gradient(ellipse 60% 70% at 85% 75%, var(--mesh-2) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 50% 10%, var(--mesh-3) 0%, transparent 55%);
}

.bg-dots {
    position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background-image: radial-gradient(var(--dot-bg) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: radial-gradient(ellipse at center, black 20%, transparent 65%);
    transition: background 0.4s;
}

.container { max-width: 980px; margin: 0 auto; padding: 40px 24px 24px; }

/* ====== THEME TOGGLE ====== */
.theme-toggle-wrap {
    position: fixed; top: 20px; right: 24px; z-index: 100;
}

.theme-toggle {
    width: 56px; height: 30px;
    background: var(--toggle-bg);
    border-radius: 100px;
    border: 1px solid var(--border);
    cursor: pointer;
    position: relative;
    transition: all 0.35s ease;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 7px;
    font-size: 13px;
}

.theme-toggle::after {
    content: '';
    position: absolute;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: var(--toggle-dot);
    top: 3px; left: 3px;
    transition: all 0.35s cubic-bezier(0.68, -0.2, 0.27, 1.2);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

[data-theme="dark"] .theme-toggle::after {
    left: 29px;
    background: var(--accent);
    box-shadow: 0 2px 10px rgba(129,140,248,0.4);
}

/* ====== HEADER ====== */
.header { text-align: center; margin-bottom: 48px; }

.logo {
    width: 60px; height: 60px; margin: 0 auto 18px;
    background: var(--accent-light);
    border: 1.5px solid var(--accent-border);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: var(--shadow-md);
    transition: all 0.4s;
}

h1 { font-size: 2.4rem; font-weight: 900; letter-spacing: -1.5px; line-height: 1.15; margin-bottom: 12px; }
h1 .g { background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.sub { color: var(--text-secondary); font-size: 1rem; font-weight: 400; max-width: 560px; margin: 0 auto 20px; line-height: 1.65; }

.pills { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }

.pill {
    display:inline-flex; align-items:center; gap:5px;
    padding: 5px 14px; border-radius: 100px;
    font-size: 0.73rem; font-weight: 600;
    background: var(--accent-light);
    border: 1px solid var(--accent-border);
    color: var(--accent);
    transition: all 0.3s;
}

.pill .d { width:5px; height:5px; border-radius:50%; background: currentColor; }
.pill.em { color: var(--safe); background: var(--safe-bg); border-color: var(--safe-border); }

/* ====== INPUT CARD ====== */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 32px;
    box-shadow: var(--shadow-md);
    margin-bottom: 24px;
    position: relative;
    transition: all 0.4s;
}

.card .top-line {
    position: absolute; top: 0; left: 24px; right: 24px; height: 2px;
    background: var(--gradient); border-radius: 0 0 2px 2px; opacity: 0.5;
}

.card label {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.75rem; font-weight: 700; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px;
}

textarea {
    width: 100%; min-height: 100px;
    background: var(--bg-input);
    border: 1.5px solid var(--border);
    border-radius: 14px; padding: 16px 18px;
    color: var(--text);
    font-family: 'Inter', sans-serif; font-size: 1rem; line-height: 1.6;
    resize: vertical; transition: all 0.3s;
}

textarea:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-light);
    background: var(--bg-alt);
}

textarea::placeholder { color: var(--text-muted); }

.ctrls { display:flex; align-items:center; gap:10px; margin-top:16px; flex-wrap:wrap; }

.btn {
    padding: 12px 28px; border-radius: 12px;
    font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 700;
    cursor: pointer; border: none; transition: all 0.25s;
}

.btn-go {
    background: var(--gradient); color: #fff;
    box-shadow: 0 4px 18px rgba(99,102,241,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
}
.btn-go:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(99,102,241,0.4); }
.btn-go:active { transform: translateY(0); }
.btn-go:disabled { opacity:0.4; cursor:not-allowed; transform:none; }

.btn-x {
    background: transparent; color: var(--text-muted);
    border: 1.5px solid var(--border); padding: 11px 22px;
}
.btn-x:hover { border-color: var(--border-hover); color: var(--text-secondary); background: var(--bg-hover); }

.hint { font-size:0.7rem; color:var(--text-muted); margin-left:auto; display:flex; align-items:center; gap:4px; }
kbd { padding:2px 6px; background:var(--bg-input); border:1px solid var(--border); border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:var(--text-secondary); }

.ex-label { font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin:18px 0 9px; }
.chips { display:flex; gap:7px; flex-wrap:wrap; }

.chip {
    padding: 6px 14px; border-radius: 100px;
    font-size: 0.78rem; font-weight: 500;
    background: var(--bg-hover); border: 1px solid var(--border);
    color: var(--text-secondary); cursor: pointer;
    transition: all 0.2s;
}

.chip:hover {
    background: var(--accent-light); border-color: var(--accent-border);
    color: var(--accent); transform: translateY(-1px);
}

/* ====== LOADING ====== */
#loading { display:none; text-align:center; padding:50px 20px; }
#loading.on { display:block; }

.ld { width:44px; height:44px; margin:0 auto 16px; position:relative; }
.ld::before,.ld::after { content:''; position:absolute; inset:0; border-radius:50%; border:3px solid transparent; }
.ld::before { border-top-color: var(--accent); animation: sp 0.85s linear infinite; }
.ld::after { border-bottom-color: #a855f7; animation: sp 0.85s linear infinite reverse; inset:6px; }
@keyframes sp { to { transform:rotate(360deg); } }

.ld-t { color:var(--text-secondary); font-size:0.9rem; font-weight:500; }
.ld-s { color:var(--text-muted); font-size:0.76rem; margin-top:4px; }

/* ====== RESULTS ====== */
#results { display:none; } #results.on { display:block; animation: fu 0.45s ease; }
@keyframes fu { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

/* Hero */
.hero {
    background: var(--bg-card); border: 1.5px solid var(--border);
    border-radius: 22px; padding: 36px 32px; text-align: center;
    margin-bottom: 18px; position: relative; overflow: hidden;
    box-shadow: var(--shadow-md); transition: all 0.4s;
}

.hero .bar { position:absolute; top:0; left:0; right:0; height:3px; }
.hero.safe .bar  { background: linear-gradient(90deg, transparent, var(--safe), transparent); }
.hero.warn .bar  { background: linear-gradient(90deg, transparent, var(--warn), transparent); }
.hero.danger .bar{ background: linear-gradient(90deg, transparent, var(--danger), transparent); }
.hero.info .bar  { background: linear-gradient(90deg, transparent, var(--info), transparent); }

.h-tag {
    display:inline-flex; align-items:center; gap:5px;
    padding:5px 14px; border-radius:100px;
    font-size:0.7rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1.2px; margin-bottom:14px;
}
.h-tag.safe  { background:var(--safe-bg); color:var(--safe); border:1px solid var(--safe-border); }
.h-tag.warn  { background:var(--warn-bg); color:var(--warn); border:1px solid var(--warn-border); }
.h-tag.danger{ background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-border); }
.h-tag.info  { background:var(--info-bg); color:var(--info); border:1px solid var(--info-border); }

.h-pred { font-size:1.9rem; font-weight:900; letter-spacing:-0.5px; margin-bottom:14px; }

.h-stats { display:flex; justify-content:center; gap:32px; flex-wrap:wrap; }
.h-stat .v { font-size:1.4rem; font-weight:800; font-family:'JetBrains Mono',monospace; }
.h-stat .l { font-size:0.68rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin-top:2px; }

.votes { display:flex; gap:7px; justify-content:center; margin-top:16px; }
.vd {
    width:11px; height:11px; border-radius:50%;
    border:2px solid var(--border); transition:all 0.3s;
}
.vd.y { border-color:var(--safe); background:var(--safe); box-shadow:0 0 8px rgba(5,150,105,0.3); }
.vd.n { border-color:var(--danger); background:var(--danger); }

/* Model Grid */
.mgrid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px; }
@media(max-width:780px) { .mgrid { grid-template-columns:1fr; } }

.mc {
    background: var(--bg-card); border:1.5px solid var(--border);
    border-radius: 18px; padding: 22px; transition: all 0.3s;
    box-shadow: var(--shadow-sm);
}
.mc:hover { border-color:var(--border-hover); transform:translateY(-3px); box-shadow:var(--shadow-lg); }

.mc-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.mc-n { font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.2px; }
.mc-b { padding:3px 9px; border-radius:100px; font-size:0.62rem; font-weight:700; }

.mc-p { font-size:1.02rem; font-weight:800; margin-bottom:3px; }
.mc-c { font-size:0.8rem; color:var(--text-secondary); margin-bottom:16px; font-family:'JetBrains Mono',monospace; font-weight:500; }

.pr { display:flex; align-items:center; gap:7px; margin-bottom:7px; }
.pr-n { font-size:0.63rem; color:var(--text-muted); width:75px; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex-shrink:0; }
.pr-t { flex:1; height:5px; background:var(--bg-input); border-radius:3px; overflow:hidden; }
.pr-f { height:100%; border-radius:3px; transition: width 0.7s cubic-bezier(0.16,1,0.3,1); }
.pr-v { font-size:0.62rem; font-family:'JetBrains Mono',monospace; color:var(--text-muted); width:36px; text-align:right; }

/* Preprocessed */
.ppbox {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:14px; padding:16px 20px;
    display:flex; align-items:flex-start; gap:10px;
    box-shadow:var(--shadow-sm); transition:all 0.4s;
}
.ppbox .ic { font-size:15px; color:var(--text-muted); margin-top:2px; }
.pp-l { font-size:0.65rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }
.pp-t { font-size:0.85rem; color:var(--text-secondary); font-family:'JetBrains Mono',monospace; word-break:break-word; }

/* Footer */
footer { text-align:center; padding:40px 0 24px; color:var(--text-muted); font-size:0.76rem; }
footer a { color:var(--accent); text-decoration:none; }
footer a:hover { text-decoration:underline; }
.fstats { display:flex; gap:20px; justify-content:center; margin-bottom:10px; flex-wrap:wrap; }
.fstats strong { color:var(--text-secondary); font-weight:600; }
</style>
</head>
<body>
<div class="bg-decor"></div>
<div class="bg-dots"></div>

<div class="theme-toggle-wrap">
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
        <span>☀️</span><span>🌙</span>
    </div>
</div>

<div class="container">
    <div class="header">
        <div class="logo">🛡️</div>
        <h1><span class="g">Tamil Offensive Language</span><br>Detector</h1>
        <p class="sub">Real-time offensive language detection for Tamil, Tanglish & code-mixed text using an ensemble of three multilingual transformer models.</p>
        <div class="pills">
            <span class="pill"><span class="d"></span>MuRIL · 236M</span>
            <span class="pill"><span class="d"></span>XLM-RoBERTa · 278M</span>
            <span class="pill"><span class="d"></span>mBERT · 177M</span>
            <span class="pill em"><span class="d"></span>Ensemble F1: 0.762</span>
        </div>
    </div>

    <div class="card">
        <div class="top-line"></div>
        <label>✍️ Enter Tamil / Tanglish / Code-Mixed Text</label>
        <textarea id="inp" placeholder="Type or paste text here... e.g., 'Padam vera level mass iruku'"></textarea>
        <div class="ctrls">
            <button class="btn btn-go" id="goBtn" onclick="analyze()">⚡ Analyze</button>
            <button class="btn btn-x" onclick="clearAll()">Clear</button>
            <span class="hint"><kbd>Enter</kbd> to analyze</span>
        </div>
        <div class="ex-label">Try these examples</div>
        <div class="chips">
            <span class="chip" onclick="setEx(this)">Padam vera level mass iruku</span>
            <span class="chip" onclick="setEx(this)">ithellam oru padama da</span>
            <span class="chip" onclick="setEx(this)">nalla iruku bro</span>
            <span class="chip" onclick="setEx(this)">ivan oru waste fellow</span>
            <span class="chip" onclick="setEx(this)">super acting in this movie</span>
            <span class="chip" onclick="setEx(this)">dei romba mokka da</span>
        </div>
    </div>

    <div id="loading">
        <div class="ld"></div>
        <div class="ld-t">Analyzing with 3 models…</div>
        <div class="ld-s">MuRIL · XLM-RoBERTa · mBERT</div>
    </div>

    <div id="results">
        <div class="hero" id="hero"></div>
        <div class="mgrid" id="mgrid"></div>
        <div class="ppbox" id="ppbox"></div>
    </div>

    <footer>
        <div class="fstats">
            <span>Dataset: <strong>35,138 samples</strong></span>
            <span>Classes: <strong>6</strong></span>
            <span>Models: <strong>3 + ensemble</strong></span>
        </div>
        Built for research · <a href="https://github.com/naveen-231006/Hate-Word" target="_blank">View on GitHub ↗</a>
    </footer>
</div>

<script>
const C={Not_offensive:'#059669',Offensive_Untargeted:'#d97706',Offensive_Targeted_Individual:'#dc2626',
    Offensive_Targeted_Group:'#b91c1c',Offensive_Targeted_Other:'#7f1d1d','not-Tamil':'#4f46e5'};
const CD={Not_offensive:'#34d399',Offensive_Untargeted:'#fbbf24',Offensive_Targeted_Individual:'#f87171',
    Offensive_Targeted_Group:'#ef4444',Offensive_Targeted_Other:'#991b1b','not-Tamil':'#818cf8'};
const E={Not_offensive:'✅',Offensive_Untargeted:'⚠️',Offensive_Targeted_Individual:'🎯',
    Offensive_Targeted_Group:'👥',Offensive_Targeted_Other:'❗','not-Tamil':'🌐'};
const S={Not_offensive:'safe',Offensive_Untargeted:'warn',Offensive_Targeted_Individual:'danger',
    Offensive_Targeted_Group:'danger',Offensive_Targeted_Other:'danger','not-Tamil':'info'};
const N={Not_offensive:'Not Offensive',Offensive_Untargeted:'Offensive (Untargeted)',
    Offensive_Targeted_Individual:'Targeted: Individual',Offensive_Targeted_Group:'Targeted: Group',
    Offensive_Targeted_Other:'Targeted: Other','not-Tamil':'Not Tamil'};

function isDark(){return document.documentElement.getAttribute('data-theme')==='dark'}
function getColor(label){return isDark()?CD[label]:C[label]}

function toggleTheme(){
    const t=isDark()?'light':'dark';
    document.documentElement.setAttribute('data-theme',t);
    localStorage.setItem('theme',t);
}
(function(){const t=localStorage.getItem('theme')||'light';document.documentElement.setAttribute('data-theme',t);})();

function setEx(el){document.getElementById('inp').value=el.textContent;document.getElementById('inp').focus();}
function clearAll(){document.getElementById('inp').value='';document.getElementById('results').classList.remove('on');document.getElementById('inp').focus();}

async function analyze(){
    const text=document.getElementById('inp').value.trim();
    if(!text)return;
    const btn=document.getElementById('goBtn');
    btn.disabled=true;btn.textContent='⏳ Analyzing…';
    document.getElementById('loading').classList.add('on');
    document.getElementById('results').classList.remove('on');
    try{
        const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
        const d=await r.json();
        render(d);
    }catch(e){alert('Error: '+e.message);}
    finally{btn.disabled=false;btn.textContent='⚡ Analyze';document.getElementById('loading').classList.remove('on');}
}

function render(data){
    const{results,cleaned_text,vote_count}=data;
    const ens=results.ensemble;
    const sev=S[ens.prediction];
    const col=getColor(ens.prediction);

    document.getElementById('hero').className='hero '+sev;
    document.getElementById('hero').innerHTML=`
        <div class="bar"></div>
        <span class="h-tag ${sev}">${E[ens.prediction]} Ensemble Prediction</span>
        <div class="h-pred" style="color:${col}">${N[ens.prediction]}</div>
        <div class="h-stats">
            <div class="h-stat"><div class="v" style="color:${col}">${(ens.confidence*100).toFixed(1)}%</div><div class="l">Avg Confidence</div></div>
            <div class="h-stat"><div class="v">${vote_count}/3</div><div class="l">Models Agree</div></div>
        </div>
        <div class="votes">
            ${['mbert','muril','xlm-roberta'].map(k=>`<span class="vd ${results[k].prediction===ens.prediction?'y':'n'}" title="${results[k].model}: ${N[results[k].prediction]}"></span>`).join('')}
        </div>`;

    const grid=document.getElementById('mgrid');grid.innerHTML='';
    for(const key of['mbert','muril','xlm-roberta']){
        const r=results[key];const c=getColor(r.prediction);const ag=r.prediction===ens.prediction;
        let bars='';
        Object.entries(r.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([l,p])=>{
            const pct=(p*100).toFixed(1);
            bars+=`<div class="pr"><span class="pr-n">${N[l].split(':').pop().trim()}</span><div class="pr-t"><div class="pr-f" style="width:${pct}%;background:${getColor(l)}"></div></div><span class="pr-v">${pct}%</span></div>`;
        });
        grid.innerHTML+=`<div class="mc"><div class="mc-top"><span class="mc-n">${r.model}</span><span class="mc-b" style="background:${ag?'var(--safe-bg)':'var(--danger-bg)'};color:${ag?'var(--safe)':'var(--danger)'};border:1px solid ${ag?'var(--safe-border)':'var(--danger-border)'}">${ag?'✓ Agree':'✗ Differ'}</span></div><div class="mc-p" style="color:${c}">${E[r.prediction]} ${N[r.prediction]}</div><div class="mc-c">${(r.confidence*100).toFixed(1)}% confidence</div>${bars}</div>`;
    }

    document.getElementById('ppbox').innerHTML=`<span class="ic">🔧</span><div><div class="pp-l">Preprocessed Input</div><div class="pp-t">${cleaned_text}</div></div>`;
    document.getElementById('results').classList.add('on');
}

document.getElementById('inp').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();analyze();}});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    if not text: return jsonify({'error': 'No text provided'}), 400
    return jsonify(predict(text))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Tamil Offensive Language Detector")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
