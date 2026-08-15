import re, math, json
from collections import Counter

TRANSITION_WORDS = ["moreover","furthermore","additionally","however","therefore","thus",
 "in conclusion","it is important to note","it is a testament","testament to",
 "underscore","underscored","delve","boundless","tapestry","navigate","navigating",
 "profound","profoundly","invaluable","unwavering","paramount","cannot be overstated",
 "plays a pivotal role","serves as a","this experience taught me","this journey",
 "not only","but also"]

CONTRACTIONS = re.compile(r"\b\w+'(m|re|ve|ll|d|s|t)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])\s*\n+')

def split_sentences(text):
    text = text.strip()
    paras = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    sents = []
    for p in paras:
        parts = SENT_SPLIT.split(p.strip())
        for s in parts:
            s = s.strip()
            if s:
                sents.append(s)
    return sents

def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]

def lexicon_hits(text_lower):
    return sum(text_lower.count(t) for t in TRANSITION_WORDS)

def punct_counts(text):
    return Counter(c for c in text if c in ".,;:!?\u2014-")

def sentence_features(sent, essay_sent_lengths_mean, essay_sent_lengths_std, ngram):
    w = words(sent)
    n = len(w)
    tl = sent.lower()
    lp = ngram.avg_logprob(w) if n > 0 else 0.0
    length_z = 0.0
    if essay_sent_lengths_std > 1e-6:
        length_z = (n - essay_sent_lengths_mean) / essay_sent_lengths_std
    lex = lexicon_hits(tl)
    lex_rate = lex / max(n, 1) * 100
    contractions = len(CONTRACTIONS.findall(sent))
    contraction_rate = contractions / max(n, 1) * 100
    pc = punct_counts(sent)
    punct_types = len([k for k in pc if pc[k] > 0])
    avg_word_len = sum(len(x) for x in w) / max(n, 1)
    ttr = len(set(w)) / max(n, 1)
    return {
        "n_words": n,
        "avg_logprob": lp,
        "length_z": length_z,
        "lex_rate": lex_rate,
        "contraction_rate": contraction_rate,
        "punct_types": punct_types,
        "avg_word_len": avg_word_len,
        "ttr": ttr,
        "lex_hits": lex,
    }

def essay_features(text, ngram):
    sents = split_sentences(text)
    lens = [len(words(s)) for s in sents]
    mean_len = sum(lens) / max(len(lens), 1)
    var_len = sum((l - mean_len) ** 2 for l in lens) / max(len(lens), 1)
    std_len = math.sqrt(var_len)

    logprobs = []
    lex_total = 0
    contraction_total = 0
    total_words = 0
    punct_set = set()
    starters = []
    for s in sents:
        w = words(s)
        total_words += len(w)
        if w:
            logprobs.append(ngram.avg_logprob(w))
            starters.append(w[0])
        lex_total += lexicon_hits(s.lower())
        contraction_total += len(CONTRACTIONS.findall(s))
        for c in s:
            if c in ".,;:!?\u2014-":
                punct_set.add(c)

    mean_lp = sum(logprobs) / max(len(logprobs), 1)
    var_lp = sum((x - mean_lp) ** 2 for x in logprobs) / max(len(logprobs), 1)
    std_lp = math.sqrt(var_lp)  # burstiness: LOWER std = more uniform = more AI-like

    starter_counts = Counter(starters)
    starter_repeat_rate = 0.0
    if starters:
        top = starter_counts.most_common(1)[0][1]
        starter_repeat_rate = top / len(starters)

    allw = words(text)
    ttr = len(set(allw)) / max(len(allw), 1)

    # repeated trigram phrases (self-repetition of exact 3-word sequences)
    trigrams = [tuple(allw[i:i+3]) for i in range(len(allw)-2)]
    tri_counts = Counter(trigrams)
    repeated_trigrams = sum(1 for c in tri_counts.values() if c > 1)
    repeated_trigram_rate = repeated_trigrams / max(len(trigrams), 1) * 100

    return {
        "mean_sent_len": mean_len,
        "std_sent_len": std_len,
        "mean_logprob": mean_lp,
        "std_logprob": std_lp,
        "lex_rate_per100w": lex_total / max(total_words, 1) * 100,
        "contraction_rate_per100w": contraction_total / max(total_words, 1) * 100,
        "punct_variety": len(punct_set),
        "ttr": ttr,
        "starter_repeat_rate": starter_repeat_rate,
        "repeated_trigram_rate": repeated_trigram_rate,
        "n_sentences": len(sents),
        "n_words": total_words,
    }, sents, lens, logprobs
