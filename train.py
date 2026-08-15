import json, math, random
import numpy as np
from corpus import HUMAN_TRAIN, AI_TRAIN, HUMAN_TEST, AI_TEST, ESL_TEST
from features import essay_features, words, split_sentences
from ngram import NgramModel

random.seed(7)

# 1. Train n-gram LM on human-written text only (reference "normal human prose")
ngram = NgramModel(k=0.6)
sent_tokens = []
for essay in HUMAN_TRAIN:
    for s in split_sentences(essay):
        sent_tokens.append(words(s))
ngram.fit(sent_tokens)

FEATURE_NAMES = ["mean_sent_len","std_sent_len","mean_logprob","std_logprob",
    "lex_rate_per100w","contraction_rate_per100w","punct_variety","ttr",
    "starter_repeat_rate","repeated_trigram_rate"]

def vec(text):
    f, *_ = essay_features(text, ngram)
    return [f[name] for name in FEATURE_NAMES], f

X_train, y_train, raw_train = [], [], []
for e in HUMAN_TRAIN:
    v, f = vec(e); X_train.append(v); y_train.append(0); raw_train.append(("human", e, f))
for e in AI_TRAIN:
    v, f = vec(e); X_train.append(v); y_train.append(1); raw_train.append(("ai", e, f))

X_train = np.array(X_train); y_train = np.array(y_train)
mu = X_train.mean(axis=0); sigma = X_train.std(axis=0); sigma[sigma < 1e-8] = 1.0
Xz = (X_train - mu) / sigma

# 2. Logistic regression via simple gradient descent (transparent, no black-box lib needed,
# but let's just use it correctly with L2 reg)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

n, d = Xz.shape
w = np.zeros(d); b = 0.0
lr = 0.3; l2 = 0.05
for epoch in range(3000):
    z = Xz @ w + b
    p = sigmoid(z)
    grad_w = Xz.T @ (p - y_train) / n + l2 * w
    grad_b = np.mean(p - y_train)
    w -= lr * grad_w
    b -= lr * grad_b

def predict_proba(text):
    v, f = vec(text)
    vz = (np.array(v) - mu) / sigma
    p = sigmoid(vz @ w + b)
    return float(p), f, v

# 3. Honest evaluation on held-out test set
results = []
correct = 0
for label, group in [("human", HUMAN_TEST), ("ai", AI_TEST)]:
    y_true = 0 if label == "human" else 1
    for e in group:
        p, f, v = predict_proba(e)
        pred = 1 if p >= 0.5 else 0
        ok = (pred == y_true)
        correct += ok
        results.append({"label": label, "prob_ai": p, "correct": ok, "text": e[:80]+"...", "features": f})

acc = correct / len(results)

# ESL check: what fraction get flagged >0.5 as AI (false positive risk on non-native English)
esl_results = []
for e in ESL_TEST:
    p, f, v = predict_proba(e)
    esl_results.append({"prob_ai": p, "text": e[:80]+"..."})

# find 3 most confidently WRONG on test set
wrong = [r for r in results if not r["correct"]]
wrong_sorted = sorted(wrong, key=lambda r: abs(r["prob_ai"] - (0 if r["label"]=="human" else 1)), reverse=True)

print("=== TEST ACCURACY:", acc, f"({correct}/{len(results)}) ===")
for r in results:
    print(f"[{r['label']:5s}] p_ai={r['prob_ai']:.3f} correct={r['correct']}  {r['text']}")

print("\n=== ESL SET (want these to score LOW/uncertain, not confidently AI) ===")
for r in esl_results:
    print(f"p_ai={r['prob_ai']:.3f}  {r['text']}")

print(f"\n=== {min(3,len(wrong_sorted))} MOST CONFIDENTLY WRONG ===")
for r in wrong_sorted[:3]:
    print(json.dumps(r, indent=2)[:800])

# 4. Export everything the JS app needs
export = {
    "feature_names": FEATURE_NAMES,
    "mu": mu.tolist(),
    "sigma": sigma.tolist(),
    "weights": w.tolist(),
    "bias": float(b),
    "ngram": ngram.to_json_compatible(top_n=8000),
    "eval": {
        "accuracy": acc,
        "n_test": len(results),
        "correct": int(correct),
        "results": [{"label": r["label"], "prob_ai": r["prob_ai"], "correct": bool(r["correct"])} for r in results],
        "esl": esl_results,
    }
}
with open("model_export.json", "w") as fp:
    json.dump(export, fp)

print("\nExported model_export.json, size (KB):",
      len(json.dumps(export)) / 1024)
