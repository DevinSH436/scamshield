"""ScamShield: an explainable scam/phishing message detector built from scratch.

A binarized multinomial Naive Bayes classifier with Laplace smoothing, written in pure
Python (no dependencies). Besides a verdict, it explains *which words* pushed
the message toward "scam" or "safe", so users learn the warning signs instead
of blindly trusting a black box.

Usage:
    python scamshield.py                      # evaluate + interactive mode
    python scamshield.py "Your package is held, pay $1.99 at http://bit.ly/x"
"""
from __future__ import annotations

import csv
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "messages.csv"

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "it",
    "at", "be", "this", "that", "with", "i", "we", "you", "me", "my", "are",
    "was", "will", "so", "do", "if", "as", "by", "from", "have", "has", "just",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, normalise risky patterns into feature tokens, split words."""
    text = text.lower()
    feats = []
    if re.search(r"https?://|www\.|bit\.ly|tinyurl", text):
        feats.append("__url__")
    if re.search(r"[$£€]\s?\d", text):
        feats.append("__money__")
    if re.search(r"\b\d{4,}\b", text):
        feats.append("__long_number__")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    words = re.findall(r"[a-z']+", text)
    return feats + [w for w in words if w not in STOPWORDS and len(w) > 1]


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.word_counts: dict[str, Counter] = {}
        self.class_counts: Counter = Counter()
        self.total_words: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayes":
        for text, label in zip(texts, labels):
            # Binarized NB: count each word once per message. This works better
            # for short texts, where repetition says little.
            tokens = set(tokenize(text))
            self.class_counts[label] += 1
            self.word_counts.setdefault(label, Counter()).update(tokens)
            self.total_words[label] += len(tokens)
            self.vocab.update(tokens)
        return self

    def _log_likelihood(self, word: str, label: str) -> float:
        count = self.word_counts[label][word]
        return math.log((count + self.alpha) /
                        (self.total_words[label] + self.alpha * len(self.vocab)))

    def predict_proba(self, text: str) -> dict[str, float]:
        n = sum(self.class_counts.values())
        tokens = {t for t in tokenize(text) if t in self.vocab}
        scores = {
            c: math.log(self.class_counts[c] / n)
            + sum(self._log_likelihood(t, c) for t in tokens)
            for c in self.class_counts
        }
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    def predict(self, text: str) -> str:
        proba = self.predict_proba(text)
        return max(proba, key=proba.get)

    def explain(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Top tokens by log-odds toward 'scam' (positive) vs 'safe' (negative)."""
        seen = dict.fromkeys(t for t in tokenize(text) if t in self.vocab)
        weights = [(t, self._log_likelihood(t, "scam") - self._log_likelihood(t, "safe"))
                   for t in seen]
        return sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:k]


def load_data(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def evaluate(texts, labels, folds: int = 5, seed: int = 42) -> dict[str, float]:
    """k-fold cross-validation; returns accuracy, precision, recall for 'scam'."""
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    tp = fp = fn = tn = 0
    for f in range(folds):
        test = set(idx[f::folds])
        train = [i for i in idx if i not in test]
        model = NaiveBayes().fit([texts[i] for i in train], [labels[i] for i in train])
        for i in test:
            pred, true = model.predict(texts[i]), labels[i]
            tp += pred == "scam" and true == "scam"
            fp += pred == "scam" and true == "safe"
            fn += pred == "safe" and true == "scam"
            tn += pred == "safe" and true == "safe"
    total = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def report(model: NaiveBayes, text: str) -> str:
    p = model.predict_proba(text)["scam"]
    verdict = "LIKELY SCAM" if p >= 0.5 else "Looks safe"
    lines = [f"{verdict}  (scam probability: {p:.0%})", "Why:"]
    for tok, w in model.explain(text):
        name = {"__url__": "[contains a link]", "__money__": "[mentions money]",
                "__long_number__": "[long number/code]"}.get(tok, f"'{tok}'")
        lines.append(f"  {name:22} {'-> scam' if w > 0 else '-> safe'} ({w:+.2f})")
    return "\n".join(lines)


def main() -> None:
    texts, labels = load_data()
    model = NaiveBayes().fit(texts, labels)
    if len(sys.argv) > 1:
        print(report(model, " ".join(sys.argv[1:])))
        return
    m = evaluate(texts, labels)
    print(f"5-fold CV on {len(texts)} messages: accuracy {m['accuracy']:.0%}, "
          f"precision {m['precision']:.0%}, recall {m['recall']:.0%}\n")
    print("Paste a message to check it (empty line to quit).")
    while (msg := input("> ").strip()):
        print(report(model, msg), "\n")


if __name__ == "__main__":
    main()
