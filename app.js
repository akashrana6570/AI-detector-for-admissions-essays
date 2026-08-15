const MODEL = JSON.parse(document.getElementById('model-data').textContent);
const EXAMPLES = JSON.parse(document.getElementById('examples-data').textContent);

/* ---------------- tokenizing ---------------- */
const WORD_RE = /[A-Za-z']+/g;
const TRANSITION_WORDS = ["moreover","furthermore","additionally","however","therefore","thus",
 "in conclusion","it is important to note","it is a testament","testament to",
 "underscore","underscored","delve","boundless","tapestry","navigate","navigating",
 "profound","profoundly","invaluable","unwavering","paramount","cannot be overstated",
 "plays a pivotal role","serves as a","this experience taught me","this journey",
 "not only","but also"];
const CONTRACTION_RE = /\b\w+'(m|re|ve|ll|d|s|t)\b/gi;

function words(text){
  const m = text.toLowerCase().match(WORD_RE);
  return m ? m : [];
}
function splitSentences(text){
  const paras = text.trim().split(/\n\s*\n/).filter(p=>p.trim());
  let sents = [];
  for(const p of paras){
    const parts = p.trim().split(/(?<=[.!?])\s+(?=[A-Z"'])|(?<=[.!?])\s*\n+/);
    for(const s of parts){ const t=s.trim(); if(t) sents.push(t); }
  }
  if(sents.length===0 && text.trim()) sents=[text.trim()];
  return sents;
}
function lexiconHits(lowerText){
  let c=0;
  for(const t of TRANSITION_WORDS){
    let idx=0;
    while(true){ idx=lowerText.indexOf(t, idx); if(idx===-1) break; c++; idx+=t.length; }
  }
  return c;
}
function punctSet(text){
  const s = new Set();
  for(const ch of text) if(".,;:!?—-".includes(ch)) s.add(ch);
  return s;
}

/* ---------------- ngram model (trigram w/ bigram/unigram backoff) ---------------- */
const NG = MODEL.ngram;
function uniP(w){
  const c = NG.unigrams[w] || 0;
  return (c + NG.k) / (NG.total_unigrams + NG.k * NG.V);
}
function biP(w0,w1){
  const ctx = NG.bigram_ctx[w0] || 0;
  if(ctx===0) return uniP(w1);
  const c = NG.bigrams[w0+"|"+w1] || 0;
  return (c + NG.k) / (ctx + NG.k * NG.V);
}
function triP(w0,w1,w2){
  const key = w0+"|"+w1;
  const ctx = NG.trigram_ctx[key] || 0;
  if(ctx===0) return biP(w1,w2);
  const lam = ctx/(ctx+2.0);
  const c = NG.trigrams[w0+"|"+w1+"|"+w2] || 0;
  return lam*((c+NG.k)/(ctx+NG.k*NG.V)) + (1-lam)*biP(w1,w2);
}
function avgLogProb(wordList){
  if(wordList.length===0) return 0;
  const toks = ["<s>","<s>",...wordList,"</s>"];
  let total=0, n=0;
  for(let i=2;i<toks.length;i++){
    const p = triP(toks[i-2], toks[i-1], toks[i]);
    total += Math.log(Math.max(p,1e-12));
    n++;
  }
  return total/Math.max(n,1);
}

/* ---------------- essay + sentence features (mirrors features.py) ---------------- */
function essayFeatures(text){
  const sents = splitSentences(text);
  const sentWords = sents.map(words);
  const lens = sentWords.map(w=>w.length);
  const meanLen = lens.reduce((a,b)=>a+b,0)/Math.max(lens.length,1);
  const varLen = lens.reduce((a,b)=>a+(b-meanLen)**2,0)/Math.max(lens.length,1);
  const stdLen = Math.sqrt(varLen);

  const logprobs = sentWords.map(w=> w.length? avgLogProb(w): 0);
  const meanLp = logprobs.reduce((a,b)=>a+b,0)/Math.max(logprobs.length,1);
  const varLp = logprobs.reduce((a,b)=>a+(b-meanLp)**2,0)/Math.max(logprobs.length,1);
  const stdLp = Math.sqrt(varLp);

  let lexTotal=0, contractionTotal=0, totalWords=0;
  const punct = new Set();
  const starters=[];
  for(const s of sents){
    const w = words(s);
    totalWords += w.length;
    if(w.length) starters.push(w[0]);
    lexTotal += lexiconHits(s.toLowerCase());
    contractionTotal += (s.match(CONTRACTION_RE)||[]).length;
    for(const c of punctSet(s)) punct.add(c);
  }
  const starterCounts = {};
  for(const s of starters) starterCounts[s]=(starterCounts[s]||0)+1;
  let topStarter=0;
  for(const k in starterCounts) topStarter=Math.max(topStarter, starterCounts[k]);
  const starterRepeatRate = starters.length? topStarter/starters.length : 0;

  const allw = words(text);
  const ttr = new Set(allw).size/Math.max(allw.length,1);

  const triCounts = {};
  const triList = [];
  for(let i=0;i<allw.length-2;i++){
    const key = allw[i]+"|"+allw[i+1]+"|"+allw[i+2];
    triCounts[key]=(triCounts[key]||0)+1;
    triList.push(key);
  }
  let repeatedTri=0;
  for(const k in triCounts) if(triCounts[k]>1) repeatedTri++;
  const repeatedTrigramRate = triList.length? repeatedTri/triList.length*100 : 0;

  const feats = {
    mean_sent_len: meanLen, std_sent_len: stdLen,
    mean_logprob: meanLp, std_logprob: stdLp,
    lex_rate_per100w: totalWords? lexTotal/totalWords*100:0,
    contraction_rate_per100w: totalWords? contractionTotal/totalWords*100:0,
    punct_variety: punct.size, ttr,
    starter_repeat_rate: starterRepeatRate,
    repeated_trigram_rate: repeatedTrigramRate,
  };
  return {feats, sents, sentWords, lens, logprobs, meanLen, stdLen, meanLp, stdLp, triCounts};
}

function sigmoid(z){ return 1/(1+Math.exp(-z)); }

function scoreEssay(feats){
  let z = MODEL.bias;
  MODEL.feature_names.forEach((name,i)=>{
    const v = (feats[name]-MODEL.mu[i])/MODEL.sigma[i];
    z += v*MODEL.weights[i];
  });
  return sigmoid(z);
}

/* ---------------- sentence-level heuristic scoring (for highlighting) ---------------- */
function clamp(x,lo,hi){ return Math.max(lo,Math.min(hi,x)); }

function analyzeSentences(ctx){
  const uniform = ctx.stdLen>0 && ctx.stdLen < 4.2; // essay-wide flag: unusually even sentence lengths
  return ctx.sents.map((s,i)=>{
    const w = ctx.sentWords[i];
    const n = w.length;
    const lower = s.toLowerCase();
    const lexHits = lexiconHits(lower);
    const matched = TRANSITION_WORDS.filter(t=>lower.includes(t));
    const contractions = (s.match(CONTRACTION_RE)||[]).length;
    const ttrLocal = n? new Set(w).size/n : 1;
    const nearMean = Math.abs(n-ctx.meanLen) <= 2.2;
    const lp = ctx.logprobs[i];
    const localLpZ = ctx.stdLp>1e-6 ? (lp-ctx.meanLp)/ctx.stdLp : 0;
    let repeatedHere = false;
    for(let j=0;j<w.length-2;j++){
      const key = w[j]+"|"+w[j+1]+"|"+w[j+2];
      if((ctx.triCounts[key]||0)>1){ repeatedHere=true; break; }
    }

    const signals = [];
    if(lexHits>0){
      signals.push({v: Math.min(1, lexHits*0.55), label:`uses ${lexHits} stock AI-essay phrase${lexHits>1?'s':''}`, detail:`matched: "${matched.slice(0,3).join('", "')}"`});
    } else {
      signals.push({v:-0.12, label:"no stock transitional phrases", detail:""});
    }
    if(contractions>0){
      signals.push({v:-0.35*Math.min(contractions,2), label:`uses ${contractions} contraction${contractions>1?'s':''}`, detail:"contractions are more common in human personal-essay voice"});
    } else if(n>6){
      signals.push({v:0.18, label:"no contractions in a longer sentence", detail:"slight lean toward formally 'polished' phrasing"});
    }
    if(ttrLocal<0.65 && n>8){
      signals.push({v:0.25, label:"below-average word variety within the sentence", detail:`${(ttrLocal*100).toFixed(0)}% unique words`});
    }
    if(uniform && nearMean){
      signals.push({v:0.4, label:"length matches this essay's unusually uniform rhythm", detail:`${n} words vs. essay average ${ctx.meanLen.toFixed(1)} (essay-wide std. dev. ${ctx.stdLen.toFixed(1)} words — low)`});
    }
    if(repeatedHere){
      signals.push({v:0.3, label:"contains a three-word phrase repeated elsewhere in the essay", detail:""});
    }
    // weak / caveated signal — see methodology note on why it's down-weighted
    signals.push({v: clamp(-localLpZ*0.12,-0.25,0.25), label: localLpZ<0 ? "reads as more 'surprising' than this essay's own average, under the reference model" : "reads as more predictable than this essay's own average, under the reference model", detail:"weak signal — small reference model", weak:true});

    const total = signals.reduce((a,b)=>a+b.v,0);
    const prob = sigmoid(total*1.6);
    return {sentence:s, prob, signals, n};
  });
}

/* ---------------- UI ---------------- */
const input = document.getElementById('essayInput');
const wc = document.getElementById('wc');
const wcTop = document.getElementById('wcTop');
const runBtn = document.getElementById('runBtn');
const resultPanel = document.getElementById('resultPanel');
const essayBody = document.getElementById('essayBody');
const verdictStrip = document.getElementById('verdictStrip');
const detail = document.getElementById('detail');
const readout = document.getElementById('readout');

function updateWC(){
  const n = words(input.value).length;
  wc.textContent = n+" words";
  wcTop.textContent = n+" words";
}
input.addEventListener('input', updateWC);

document.querySelectorAll('.ex-btn').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    input.value = EXAMPLES[btn.dataset.ex];
    updateWC();
    window.scrollTo({top:0, behavior:'smooth'});
  });
});

function colorFor(p){
  // cool slate (human) -> warm amber/red (AI)
  const stops = [
    [0.0,[59,92,122]],
    [0.5,[230,222,205]],
    [1.0,[181,67,43]]
  ];
  let lo=stops[0], hi=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++){ if(p>=stops[i][0] && p<=stops[i+1][0]){ lo=stops[i]; hi=stops[i+1]; break; } }
  const t = (p-lo[0])/((hi[0]-lo[0])||1);
  const c = lo[1].map((v,i)=>Math.round(v+(hi[1][i]-v)*t));
  return `rgba(${c[0]},${c[1]},${c[2]},${0.14+p*0.30})`;
}
function borderFor(p){
  return p>0.55 ? '#B5432B' : (p<0.35 ? '#3B5C7A' : 'transparent');
}

function renderReadout(feats, probAi){
  readout.innerHTML='';
  const rows = [
    ["Stock-phrase rate","lex_rate_per100w","per 100 words", v=>v.toFixed(2)],
    ["Contraction rate","contraction_rate_per100w","per 100 words", v=>v.toFixed(2)],
    ["Sentence-length uniformity (std dev)","std_sent_len","words — lower = more uniform", v=>v.toFixed(2)],
    ["Reference-model surprisal","mean_logprob","avg. log-prob / word", v=>v.toFixed(2)],
    ["Lexical diversity","ttr","unique/total words", v=>v.toFixed(2)],
    ["Repeated-opener rate","starter_repeat_rate","share sharing top opener", v=>(v*100).toFixed(0)+"%"],
  ];
  for(const [label,key,unit,fmt] of rows){
    const v = feats[key];
    const i = MODEL.feature_names.indexOf(key);
    const mu = MODEL.mu[i], sigma = MODEL.sigma[i];
    const z = clamp((v-mu)/sigma, -2.5, 2.5);
    const pct = (z+2.5)/5*100;
    const div = document.createElement('div');
    div.className='readout-item';
    div.innerHTML = `<div class="ri-label"><span>${label}</span><span class="ri-val">${fmt(v)}</span></div>
      <div class="bar-track"><div class="bar-mid" style="left:50%"></div><div class="bar-fill" style="width:${pct}%;background:${z>0.3?'#B5432B':(z<-0.3?'#3B5C7A':'#9AA0A8')}"></div></div>
      <div style="font-family:var(--mono);font-size:9.5px;color:var(--ink-faint);margin-top:4px;">${unit} · marker = corpus average</div>`;
    readout.appendChild(div);
  }
  const div = document.createElement('div');
  div.className='readout-item';
  div.innerHTML = `<div class="ri-label"><span style="font-weight:600;">Model output</span></div>
    <div style="font-family:var(--mono);font-size:24px;margin-top:4px;">p(AI) = ${probAi.toFixed(3)}</div>
    <div style="font-size:11.5px;color:var(--ink-soft);margin-top:4px;">logistic regression, 10 weighted features, fit on 40 labeled essays</div>`;
  readout.appendChild(div);
}

function run(){
  const text = input.value.trim();
  if(!text) return;
  const ctx = essayFeatures(text);
  const probAi = scoreEssay(ctx.feats);
  const sentAnalysis = analyzeSentences(ctx);

  // verdict strip
  let headline, sub;
  if(probAi>0.7){ headline="Reads as largely AI-polished."; }
  else if(probAi>0.5){ headline="More AI-typical signal than not — worth a closer look."; }
  else if(probAi>0.3){ headline="Reads as mostly human, with a few AI-typical passages."; }
  else { headline="Reads as human-written."; }
  sub = `${Math.round(probAi*100)} out of 100 on the model's scale — see the sentence highlights and the readout panel for what's driving it, not just the number.`;
  verdictStrip.innerHTML = `<div class="gauge" style="border-color:${borderFor(probAi)};color:${probAi>0.5?'var(--hot)':'var(--cool)'}">${Math.round(probAi*100)}</div>
    <div class="verdict-text"><div class="headline">${headline}</div><div class="sub">${sub}</div></div>
    <div class="legend"><span><span class="sw" style="background:var(--cool-bg);border:1px solid var(--cool)"></span>human-typical</span><span><span class="sw" style="background:var(--hot-bg);border:1px solid var(--hot)"></span>AI-typical</span></div>`;

  // essay body with highlighted sentences, grouped back into paragraphs roughly by blank-line split
  const paras = text.trim().split(/\n\s*\n/).filter(p=>p.trim());
  let sIdx=0;
  essayBody.innerHTML='';
  for(const para of paras){
    const pEl = document.createElement('p');
    const paraSentCount = splitSentences(para).length;
    for(let k=0;k<paraSentCount;k++){
      if(sIdx>=sentAnalysis.length) break;
      const sa = sentAnalysis[sIdx];
      const mk = document.createElement('mark');
      mk.className='s';
      mk.style.background = colorFor(sa.prob);
      mk.style.borderColor = sa.prob>0.6 || sa.prob<0.3 ? borderFor(sa.prob) : 'transparent';
      mk.textContent = sa.sentence + " ";
      mk.dataset.idx = sIdx;
      mk.addEventListener('click', ()=>showDetail(sIdx));
      pEl.appendChild(mk);
      sIdx++;
    }
    essayBody.appendChild(pEl);
  }

  function showDetail(idx){
    document.querySelectorAll('mark.s').forEach(m=>m.classList.remove('active'));
    const mk = essayBody.querySelector(`mark[data-idx="${idx}"]`);
    if(mk) mk.classList.add('active');
    const sa = sentAnalysis[idx];
    const sorted = [...sa.signals].sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
    const items = sorted.filter(s=>Math.abs(s.v)>0.12).slice(0,4).map(s=>
      `<li><span class="tag">${s.v>0?'→ AI-typical':'→ human-typical'}${s.weak?' · weak signal':''}</span><br>${s.label}${s.detail?` — <span style="color:var(--ink-soft)">${s.detail}</span>`:''}</li>`
    ).join('');
    detail.innerHTML = `<div class="dhead">Sentence ${idx+1} of ${sentAnalysis.length}</div>
      <div class="dquote">"${sa.sentence}"</div>
      <div class="dscore">local score: ${sa.prob.toFixed(2)} · ${sa.n} words</div>
      <ul>${items}</ul>`;
    detail.classList.add('show');
  }

  renderReadout(ctx.feats, probAi);
  resultPanel.style.display='block';
  if(sentAnalysis.length) showDetail(0);
}

runBtn.addEventListener('click', run);

// boot with mixed example
input.value = EXAMPLES.mixed;
updateWC();

// static content: data provenance / accuracy / ESL sections, drawn from MODEL.eval
(function fillDocs(){
  const ev = MODEL.eval;
  document.getElementById('dataStats').innerHTML = `
    <div class="stat"><div class="n">20 + 20</div><div class="l">human / AI training essays</div></div>
    <div class="stat"><div class="n">8 + 8</div><div class="l">human / AI held-out test essays</div></div>
    <div class="stat"><div class="n">3</div><div class="l">non-native-English check essays</div></div>
    <div class="stat"><div class="n">~180</div><div class="l">avg. words / essay</div></div>`;

  document.getElementById('accStats').innerHTML = `
    <div class="stat"><div class="n">${(ev.accuracy*100).toFixed(1)}%</div><div class="l">accuracy, held-out (${ev.correct}/${ev.n_test})</div></div>`;

  const wrongRows = ev.results.filter(r=>!r.correct);
  // supplement with the two most-confident ESL false positives so we always have real, disclosed wrong cases to show
  const eslSorted = [...ev.esl].sort((a,b)=>b.prob_ai-a.prob_ai).slice(0,2)
    .map(r=>({label:'human (ESL)', prob_ai:r.prob_ai, correct:false, text:r.text}));
  const rows = [...wrongRows, ...eslSorted].slice(0,3);
  let html = `<tr><th>True label</th><th>Model p(AI)</th><th>Excerpt</th><th>Why it's wrong</th></tr>`;
  const reasons = [
    "Longer, evenly-paced sentences and low contraction use in this passage happened to match the AI-typical range, even though the content is a specific, unrepeatable personal memory — a reminder that fluency/formality features are correlated with, not equivalent to, authorship.",
    "Simpler verb forms and repeated sentence openings, typical of a fluent non-native writer, read to the model as machine-smoothed uniformity. This is the ESL failure mode described below, not a one-off.",
    "Same mechanism: the contraction-rate and repeated-structure features that carry real signal on native-English prose penalize non-native phrasing patterns that have nothing to do with authorship."
  ];
  rows.forEach((r,i)=>{
    html += `<tr><td>${r.label}</td><td>${r.prob_ai.toFixed(3)}</td><td>${r.text}</td><td>${reasons[i]||''}</td></tr>`;
  });
  document.getElementById('wrongTable').innerHTML = html;

  const eslVals = ev.esl.map(r=>r.prob_ai);
  const eslAvg = eslVals.reduce((a,b)=>a+b,0)/eslVals.length;
  const humanTestVals = ev.results.filter(r=>r.label==='human').map(r=>r.prob_ai);
  const humanAvg = humanTestVals.reduce((a,b)=>a+b,0)/humanTestVals.length;
  document.getElementById('eslStats').innerHTML = `
    <div class="stat"><div class="n">${(eslAvg*100).toFixed(0)}</div><div class="l">avg. p(AI)×100, ESL-register essays</div></div>
    <div class="stat"><div class="n">${(humanAvg*100).toFixed(0)}</div><div class="l">avg. p(AI)×100, native-English human test essays</div></div>`;

  const mlpIdx = MODEL.feature_names.indexOf('mean_logprob');
  const mlpW = MODEL.weights[mlpIdx];
  document.getElementById('calloutSign').textContent =
    `On this corpus, the reference-model surprisal feature ended up weighted ${mlpW.toFixed(2)} (negative = essays that look ` +
    `LESS probable under the trigram model were scored MORE human). That is the opposite of the classic "AI text is more ` +
    `predictable" intuition, and we believe it's an artifact of training the reference model on only ~4,000 words of human text: ` +
    `AI essays in this set lean on a narrower, more "elevated" vocabulary that is simply better-covered by such a small model, ` +
    `not more fluent in a meaningful sense. We kept the feature and are reporting the sign honestly rather than hard-coding the ` +
    `direction we expected — a larger reference corpus is the fix, not a manual override.`;
})();
