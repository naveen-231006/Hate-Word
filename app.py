"""
Tamil Offensive Language Detector — Interactive Web Demo
Uses fine-tuned MuRIL, XLM-RoBERTa, and mBERT with majority-voting ensemble.
"""
import os, re, torch, numpy as np
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
mdls, toks = {}, {}
for key, cfg in MODEL_CONFIGS.items():
    print(f"Loading {cfg['name']}...")
    toks[key] = AutoTokenizer.from_pretrained(cfg['path'])
    mdls[key] = AutoModelForSequenceClassification.from_pretrained(cfg['path']).to(device).eval()
    print(f"  {cfg['name']} loaded.")
print("All models loaded!")

def predict(text):
    cleaned = preprocess_text(text)
    if not cleaned: return {'error': 'Empty text'}
    results, all_preds = {}, []
    for key in MODEL_CONFIGS:
        inputs = toks[key](cleaned, return_tensors='pt', truncation=True, max_length=128, padding='max_length').to(device)
        with torch.no_grad():
            logits = mdls[key](**inputs).logits[0].cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        idx = int(np.argmax(probs))
        all_preds.append(idx)
        results[key] = {'model': MODEL_CONFIGS[key]['name'], 'prediction': LABEL_NAMES[idx],
            'confidence': float(probs[idx]),
            'probabilities': {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}}
    ens_idx = int(mode(all_preds, keepdims=False)[0])
    avg_p = np.mean([list(results[k]['probabilities'].values()) for k in results], axis=0)
    results['ensemble'] = {'model': 'Ensemble', 'prediction': LABEL_NAMES[ens_idx],
        'confidence': float(avg_p[ens_idx]),
        'probabilities': {LABEL_NAMES[i]: float(avg_p[i]) for i in range(len(LABEL_NAMES))}}
    return {'original_text': text, 'cleaned_text': cleaned, 'results': results,
            'vote_count': sum(1 for p in all_preds if p == ens_idx)}

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tamil Offensive Language Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ===== LIGHT THEME (default) ===== */
:root,[data-theme="light"]{
    --bg:#fafafa;
    --surface:#ffffff;
    --surface-2:#f5f5f5;
    --border:#e5e5e5;
    --border-focus:#3b82f6;
    --text:#171717;
    --text-2:#525252;
    --text-3:#a3a3a3;
    --blue:#2563eb;
    --blue-bg:#eff6ff;
    --blue-border:#bfdbfe;
    --green:#16a34a;
    --green-bg:#f0fdf4;
    --green-border:#bbf7d0;
    --amber:#d97706;
    --amber-bg:#fffbeb;
    --amber-border:#fde68a;
    --red:#dc2626;
    --red-bg:#fef2f2;
    --red-border:#fecaca;
    --shadow:0 1px 3px rgba(0,0,0,0.08);
    --shadow-lg:0 8px 30px rgba(0,0,0,0.08);
    --toggle-bg:#e5e5e5;
    --toggle-knob:#fff;
}

/* ===== DARK THEME ===== */
[data-theme="dark"]{
    --bg:#111111;
    --surface:#1a1a1a;
    --surface-2:#222222;
    --border:#2e2e2e;
    --border-focus:#3b82f6;
    --text:#ededed;
    --text-2:#a1a1a1;
    --text-3:#5c5c5c;
    --blue:#60a5fa;
    --blue-bg:rgba(37,99,235,0.1);
    --blue-border:rgba(59,130,246,0.25);
    --green:#4ade80;
    --green-bg:rgba(22,163,74,0.1);
    --green-border:rgba(74,222,128,0.2);
    --amber:#fbbf24;
    --amber-bg:rgba(217,119,6,0.1);
    --amber-border:rgba(251,191,35,0.2);
    --red:#f87171;
    --red-bg:rgba(220,38,38,0.1);
    --red-border:rgba(248,113,113,0.2);
    --shadow:0 1px 3px rgba(0,0,0,0.4);
    --shadow-lg:0 8px 30px rgba(0,0,0,0.4);
    --toggle-bg:#333;
    --toggle-knob:#60a5fa;
}

*{margin:0;padding:0;box-sizing:border-box;}

body{
    font-family:'Inter',-apple-system,sans-serif;
    background:var(--bg);
    color:var(--text);
    min-height:100vh;
    transition:background .3s,color .3s;
    -webkit-font-smoothing:antialiased;
}

.wrap{max-width:900px;margin:0 auto;padding:48px 20px 32px;}

/* Toggle */
.toggle-wrap{position:fixed;top:18px;right:20px;z-index:50;}
.toggle{
    width:48px;height:26px;border-radius:13px;
    background:var(--toggle-bg);border:1px solid var(--border);
    cursor:pointer;position:relative;transition:all .3s;
    display:flex;align-items:center;justify-content:space-between;
    padding:0 6px;font-size:11px;
}
.toggle::after{
    content:'';position:absolute;width:18px;height:18px;border-radius:50%;
    background:var(--toggle-knob);top:3px;left:3px;
    transition:left .3s cubic-bezier(.4,0,.2,1);
    box-shadow:0 1px 3px rgba(0,0,0,.15);
}
[data-theme="dark"] .toggle::after{left:25px;}

/* Header */
.hdr{text-align:center;margin-bottom:40px;}
.hdr h1{font-size:1.75rem;font-weight:800;letter-spacing:-.5px;color:var(--text);margin-bottom:8px;}
.hdr p{color:var(--text-2);font-size:.92rem;max-width:520px;margin:0 auto 16px;line-height:1.6;}
.tags{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;}
.tag{
    padding:4px 12px;border-radius:6px;font-size:.72rem;font-weight:600;
    background:var(--surface-2);border:1px solid var(--border);color:var(--text-2);
}
.tag.ok{background:var(--green-bg);border-color:var(--green-border);color:var(--green);}

/* Input */
.icard{
    background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:24px;box-shadow:var(--shadow);
    margin-bottom:24px;transition:all .3s;
}
.icard label{
    display:block;font-size:.72rem;font-weight:600;color:var(--text-3);
    text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;
}
textarea{
    width:100%;min-height:90px;background:var(--surface-2);
    border:1.5px solid var(--border);border-radius:8px;
    padding:14px 16px;color:var(--text);
    font-family:'Inter',sans-serif;font-size:.95rem;line-height:1.6;
    resize:vertical;transition:all .2s;
}
textarea:focus{outline:none;border-color:var(--border-focus);box-shadow:0 0 0 3px rgba(59,130,246,.12);}
textarea::placeholder{color:var(--text-3);}

.row{display:flex;align-items:center;gap:8px;margin-top:14px;flex-wrap:wrap;}

.btn{
    padding:10px 22px;border-radius:8px;font-family:'Inter',sans-serif;
    font-size:.85rem;font-weight:600;cursor:pointer;border:none;transition:all .2s;
}
.btn-p{background:var(--blue);color:#fff;}
.btn-p:hover{opacity:.9;} .btn-p:disabled{opacity:.4;cursor:not-allowed;}
.btn-s{background:transparent;color:var(--text-3);border:1.5px solid var(--border);padding:9px 18px;}
.btn-s:hover{border-color:var(--border-focus);color:var(--text-2);}

.hint{font-size:.68rem;color:var(--text-3);margin-left:auto;}
kbd{padding:1px 5px;background:var(--surface-2);border:1px solid var(--border);border-radius:3px;font-family:'IBM Plex Mono',monospace;font-size:.62rem;}

.elbl{font-size:.68rem;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.6px;margin:16px 0 8px;}
.chips{display:flex;gap:6px;flex-wrap:wrap;}
.chip{
    padding:5px 12px;border-radius:6px;font-size:.78rem;font-weight:500;
    background:var(--surface-2);border:1px solid var(--border);
    color:var(--text-2);cursor:pointer;transition:all .15s;
}
.chip:hover{border-color:var(--blue);color:var(--blue);background:var(--blue-bg);}

/* Loading */
#ld{display:none;text-align:center;padding:48px 20px;}
#ld.on{display:block;}
.spin{width:32px;height:32px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:s .7s linear infinite;margin:0 auto 14px;}
@keyframes s{to{transform:rotate(360deg)}}
.spin-t{color:var(--text-2);font-size:.88rem;}

/* Results */
#res{display:none;}#res.on{display:block;animation:up .35s ease;}
@keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

/* Hero */
.hero{
    background:var(--surface);border:1px solid var(--border);
    border-radius:12px;padding:28px 24px;text-align:center;
    margin-bottom:16px;box-shadow:var(--shadow);transition:all .3s;
    position:relative;overflow:hidden;
}
.hero .stripe{position:absolute;top:0;left:0;right:0;height:3px;}
.hero.safe .stripe{background:var(--green);}
.hero.warn .stripe{background:var(--amber);}
.hero.danger .stripe{background:var(--red);}
.hero.info .stripe{background:var(--blue);}

.htag{
    display:inline-block;padding:4px 12px;border-radius:6px;
    font-size:.68rem;font-weight:700;text-transform:uppercase;
    letter-spacing:.8px;margin-bottom:12px;
}
.htag.safe{background:var(--green-bg);color:var(--green);border:1px solid var(--green-border);}
.htag.warn{background:var(--amber-bg);color:var(--amber);border:1px solid var(--amber-border);}
.htag.danger{background:var(--red-bg);color:var(--red);border:1px solid var(--red-border);}
.htag.info{background:var(--blue-bg);color:var(--blue);border:1px solid var(--blue-border);}

.hpred{font-size:1.5rem;font-weight:800;letter-spacing:-.3px;margin-bottom:12px;}

.hrow{display:flex;justify-content:center;gap:28px;}
.hcol .hv{font-size:1.2rem;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.hcol .hl{font-size:.65rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;}

.dots{display:flex;gap:6px;justify-content:center;margin-top:14px;}
.dot{width:10px;height:10px;border-radius:50%;border:2px solid var(--border);}
.dot.y{border-color:var(--green);background:var(--green);}
.dot.n{border-color:var(--red);background:var(--red);}

/* Grid */
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;}
@media(max-width:720px){.grid{grid-template-columns:1fr;}}

.mc{
    background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:20px;box-shadow:var(--shadow);
    transition:all .2s;
}
.mc:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px);}

.mc-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.mc-n{font-size:.68rem;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;}
.mc-b{padding:2px 8px;border-radius:4px;font-size:.6rem;font-weight:700;}

.mc-p{font-size:.95rem;font-weight:700;margin-bottom:2px;}
.mc-c{font-size:.78rem;color:var(--text-2);margin-bottom:14px;font-family:'IBM Plex Mono',monospace;}

.pb{display:flex;align-items:center;gap:6px;margin-bottom:6px;}
.pb-n{font-size:.6rem;color:var(--text-3);width:70px;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pb-t{flex:1;height:4px;background:var(--surface-2);border-radius:2px;overflow:hidden;}
.pb-f{height:100%;border-radius:2px;transition:width .6s ease;}
.pb-v{font-size:.6rem;font-family:'IBM Plex Mono',monospace;color:var(--text-3);width:34px;text-align:right;}

/* Preprocess */
.pp{
    background:var(--surface);border:1px solid var(--border);
    border-radius:8px;padding:14px 18px;display:flex;gap:10px;
    align-items:flex-start;box-shadow:var(--shadow);transition:all .3s;
}
.pp-l{font-size:.62rem;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px;}
.pp-t{font-size:.82rem;color:var(--text-2);font-family:'IBM Plex Mono',monospace;word-break:break-word;}

/* Footer */
footer{text-align:center;padding:36px 0 20px;color:var(--text-3);font-size:.74rem;}
footer a{color:var(--blue);text-decoration:none;}
footer a:hover{text-decoration:underline;}
.fs{display:flex;gap:16px;justify-content:center;margin-bottom:8px;flex-wrap:wrap;font-size:.7rem;}
.fs strong{color:var(--text-2);font-weight:600;}
</style>
</head>
<body>
<div class="toggle-wrap"><div class="toggle" onclick="tog()" title="Toggle theme"><span>☀️</span><span>🌙</span></div></div>

<div class="wrap">
<div class="hdr">
    <h1>Tamil Offensive Language Detector</h1>
    <p>Detect offensive content in Tamil, Tanglish &amp; code-mixed text using three multilingual transformer models with majority-voting ensemble.</p>
    <div class="tags">
        <span class="tag">MuRIL · 236M</span>
        <span class="tag">XLM-RoBERTa · 278M</span>
        <span class="tag">mBERT · 177M</span>
        <span class="tag ok">Ensemble F1: 0.762</span>
    </div>
</div>

<div class="icard">
    <label>Enter Tamil / Tanglish / Code-Mixed Text</label>
    <textarea id="inp" placeholder="Type or paste text here..."></textarea>
    <div class="row">
        <button class="btn btn-p" id="go" onclick="run()">Analyze</button>
        <button class="btn btn-s" onclick="clr()">Clear</button>
        <span class="hint"><kbd>Enter</kbd> to analyze</span>
    </div>
    <div class="elbl">Try an example</div>
    <div class="chips">
        <span class="chip" onclick="ex(this)">Padam vera level mass iruku</span>
        <span class="chip" onclick="ex(this)">ithellam oru padama da</span>
        <span class="chip" onclick="ex(this)">nalla iruku bro</span>
        <span class="chip" onclick="ex(this)">ivan oru waste fellow</span>
        <span class="chip" onclick="ex(this)">super acting in this movie</span>
        <span class="chip" onclick="ex(this)">dei romba mokka da</span>
    </div>
</div>

<div id="ld"><div class="spin"></div><div class="spin-t">Analyzing with 3 models...</div></div>

<div id="res">
    <div class="hero" id="hero"></div>
    <div class="grid" id="grid"></div>
    <div class="pp" id="pp"></div>
</div>

<footer>
    <div class="fs"><span>Dataset: <strong>35,138</strong></span><span>Classes: <strong>6</strong></span><span>Models: <strong>3 + ensemble</strong></span></div>
    Built for research · <a href="https://github.com/naveen-231006/Hate-Word" target="_blank">GitHub</a>
</footer>
</div>

<script>
const COL_L={Not_offensive:'#16a34a',Offensive_Untargeted:'#d97706',Offensive_Targeted_Individual:'#dc2626',Offensive_Targeted_Group:'#b91c1c',Offensive_Targeted_Other:'#7f1d1d','not-Tamil':'#2563eb'};
const COL_D={Not_offensive:'#4ade80',Offensive_Untargeted:'#fbbf24',Offensive_Targeted_Individual:'#f87171',Offensive_Targeted_Group:'#ef4444',Offensive_Targeted_Other:'#fca5a5','not-Tamil':'#60a5fa'};
const EM={Not_offensive:'✅',Offensive_Untargeted:'⚠️',Offensive_Targeted_Individual:'🎯',Offensive_Targeted_Group:'👥',Offensive_Targeted_Other:'❗','not-Tamil':'🌐'};
const SEV={Not_offensive:'safe',Offensive_Untargeted:'warn',Offensive_Targeted_Individual:'danger',Offensive_Targeted_Group:'danger',Offensive_Targeted_Other:'danger','not-Tamil':'info'};
const NM={Not_offensive:'Not Offensive',Offensive_Untargeted:'Offensive (Untargeted)',Offensive_Targeted_Individual:'Targeted · Individual',Offensive_Targeted_Group:'Targeted · Group',Offensive_Targeted_Other:'Targeted · Other','not-Tamil':'Not Tamil'};

function dk(){return document.documentElement.getAttribute('data-theme')==='dark'}
function gc(l){return dk()?COL_D[l]:COL_L[l]}
function tog(){const t=dk()?'light':'dark';document.documentElement.setAttribute('data-theme',t);localStorage.setItem('theme',t);}
(function(){document.documentElement.setAttribute('data-theme',localStorage.getItem('theme')||'light');})();

function ex(e){document.getElementById('inp').value=e.textContent;document.getElementById('inp').focus();}
function clr(){document.getElementById('inp').value='';document.getElementById('res').classList.remove('on');document.getElementById('inp').focus();}

async function run(){
    const t=document.getElementById('inp').value.trim();if(!t)return;
    const b=document.getElementById('go');b.disabled=true;b.textContent='Analyzing...';
    document.getElementById('ld').classList.add('on');document.getElementById('res').classList.remove('on');
    try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});show(await r.json());}
    catch(e){alert(e.message);}
    finally{b.disabled=false;b.textContent='Analyze';document.getElementById('ld').classList.remove('on');}
}

function show(d){
    const{results:R,cleaned_text:ct,vote_count:vc}=d;
    const e=R.ensemble,s=SEV[e.prediction],c=gc(e.prediction);

    const h=document.getElementById('hero');h.className='hero '+s;
    h.innerHTML=`<div class="stripe"></div>
        <span class="htag ${s}">${EM[e.prediction]} Ensemble Result</span>
        <div class="hpred" style="color:${c}">${NM[e.prediction]}</div>
        <div class="hrow">
            <div class="hcol"><div class="hv" style="color:${c}">${(e.confidence*100).toFixed(1)}%</div><div class="hl">Avg Confidence</div></div>
            <div class="hcol"><div class="hv">${vc}/3</div><div class="hl">Models Agree</div></div>
        </div>
        <div class="dots">${['mbert','muril','xlm-roberta'].map(k=>`<span class="dot ${R[k].prediction===e.prediction?'y':'n'}" title="${R[k].model}: ${NM[R[k].prediction]}"></span>`).join('')}</div>`;

    const g=document.getElementById('grid');g.innerHTML='';
    ['mbert','muril','xlm-roberta'].forEach(k=>{
        const r=R[k],cl=gc(r.prediction),ag=r.prediction===e.prediction;
        let bars='';
        Object.entries(r.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([l,p])=>{
            const pct=(p*100).toFixed(1);
            bars+=`<div class="pb"><span class="pb-n">${NM[l].split('·').pop().trim()}</span><div class="pb-t"><div class="pb-f" style="width:${pct}%;background:${gc(l)}"></div></div><span class="pb-v">${pct}%</span></div>`;
        });
        g.innerHTML+=`<div class="mc"><div class="mc-top"><span class="mc-n">${r.model}</span><span class="mc-b" style="background:${ag?'var(--green-bg)':'var(--red-bg)'};color:${ag?'var(--green)':'var(--red)'};border:1px solid ${ag?'var(--green-border)':'var(--red-border)'}">${ag?'Agree':'Differ'}</span></div><div class="mc-p" style="color:${cl}">${EM[r.prediction]} ${NM[r.prediction]}</div><div class="mc-c">${(r.confidence*100).toFixed(1)}%</div>${bars}</div>`;
    });

    document.getElementById('pp').innerHTML=`<div><div class="pp-l">Preprocessed</div><div class="pp-t">${ct}</div></div>`;
    document.getElementById('res').classList.add('on');
}

document.getElementById('inp').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();run();}});
</script>
</body>
</html>"""

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    text = request.get_json().get('text', '')
    if not text: return jsonify({'error': 'No text'}), 400
    return jsonify(predict(text))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Tamil Offensive Language Detector")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
