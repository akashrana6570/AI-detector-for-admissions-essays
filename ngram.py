import math
from collections import Counter, defaultdict

class NgramModel:
    """Trigram-backoff word LM (Laplace-ish add-k smoothing), trained on a
    reference corpus of human writing. Used only as an INSTRUMENT to get
    per-sentence average log-probability ("surprisal") -- a real number the
    classifier can use, not a verdict. See report for why this is a small,
    imperfect model and what that implies."""

    def __init__(self, k=0.6):
        self.k = k
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.bigram_ctx = Counter()
        self.trigram_ctx = Counter()
        self.vocab = set()
        self.total_unigrams = 0

    def fit(self, tokenized_sentences):
        for w in tokenized_sentences:
            toks = ["<s>", "<s>"] + w + ["</s>"]
            for t in w:
                self.unigrams[t] += 1
                self.vocab.add(t)
                self.total_unigrams += 1
            for i in range(len(toks) - 1):
                self.bigrams[(toks[i], toks[i+1])] += 1
                self.bigram_ctx[toks[i]] += 1
            for i in range(len(toks) - 2):
                self.trigrams[(toks[i], toks[i+1], toks[i+2])] += 1
                self.trigram_ctx[(toks[i], toks[i+1])] += 1
        self.V = len(self.vocab) + 2  # +2 for <s>/</s>

    def _unigram_p(self, w):
        return (self.unigrams.get(w, 0) + self.k) / (self.total_unigrams + self.k * self.V)

    def _bigram_p(self, w0, w1):
        ctx = self.bigram_ctx.get(w0, 0)
        if ctx == 0:
            return self._unigram_p(w1)
        return (self.bigrams.get((w0, w1), 0) + self.k) / (ctx + self.k * self.V)

    def _trigram_p(self, w0, w1, w2):
        ctx = self.trigram_ctx.get((w0, w1), 0)
        if ctx == 0:
            return self._bigram_p(w1, w2)
        # interpolate trigram with bigram backoff for robustness on sparse data
        lam = ctx / (ctx + 2.0)
        return lam * ((self.trigrams.get((w0, w1, w2), 0) + self.k) / (ctx + self.k * self.V)) \
               + (1 - lam) * self._bigram_p(w1, w2)

    def token_logprob(self, w0, w1, w2):
        p = self._trigram_p(w0, w1, w2)
        return math.log(max(p, 1e-12))

    def avg_logprob(self, words):
        toks = ["<s>", "<s>"] + words + ["</s>"]
        total = 0.0
        n = 0
        for i in range(2, len(toks)):
            total += self.token_logprob(toks[i-2], toks[i-1], toks[i])
            n += 1
        return total / max(n, 1)

    def to_json_compatible(self, top_n=6000):
        """Export a compact version for the JS port: unigram counts (top_n),
        bigram counts (top_n), trigram counts (top_n), plus totals."""
        def top(counter, n):
            return {("|".join(k) if isinstance(k, tuple) else k): v
                    for k, v in counter.most_common(n)}
        return {
            "k": self.k,
            "V": self.V,
            "total_unigrams": self.total_unigrams,
            "unigrams": top(self.unigrams, top_n),
            "bigram_ctx": top(self.bigram_ctx, top_n),
            "bigrams": top(self.bigrams, top_n),
            "trigram_ctx": top(self.trigram_ctx, top_n),
            "trigrams": top(self.trigrams, top_n),
        }
