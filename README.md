# ScamShield: Explainable Scam Message Detector

Text-message and email scams (fake deliveries, "your account is locked", the "Hi mom, new number" trick) take billions of dollars a year, and they hit older and less tech-savvy people hardest. **ScamShield** is a small AI that flags suspicious messages and **explains why**, so people learn the warning signs instead of trusting a black box.

Built **from scratch in pure Python**: no scikit-learn, no external dependencies. The model is a binarized multinomial Naive Bayes classifier with Laplace smoothing.

## Demo

```text
$ python scamshield.py "Your package is held at the depot. Pay $2.50 fee here http://bit.ly/x"
LIKELY SCAM  (scam probability: 100%)
Why:
  [mentions money]       -> scam (+2.68)
  [contains a link]      -> scam (+2.39)
  'pay'                  -> scam (+1.98)
  'fee'                  -> scam (+1.29)
  'here'                 -> scam (+1.00)

$ python scamshield.py "See you at practice tomorrow"
Looks safe  (scam probability: 4%)
Why:
  'tomorrow'             -> safe (-1.20)
  'see'                  -> safe (-0.79)
  'practice'             -> safe (-0.79)
```

## How it works

1. **Feature extraction**: text is lowercased and tokenized. Risky patterns become special tokens: `[contains a link]`, `[mentions money]` and `[long number/code]`.
2. **Training**: for each class (`scam` / `safe`), the model counts how many messages contain each token (binarized counts work better on short texts).
3. **Prediction**: Bayes' rule combines the class prior with each token's smoothed likelihood in log space, then a softmax turns the scores into a probability.
4. **Explanation**: each token's log-odds `log P(token|scam) - log P(token|safe)` shows how strongly it pushed the verdict either way.

## Results

5-fold cross-validation on the bundled dataset of 85 hand-written synthetic messages:

| Metric | Score |
|---|---|
| Accuracy | 89% |
| Precision (scam) | 82% |
| Recall (scam) | 94% |

The model favors high recall on purpose: missing a scam costs more than a false alarm. The dataset is small and illustrative. Adding real labelled data (for example, the public SMS Spam Collection) would make it far more robust.

## Run it

Requires Python 3.9+ and nothing else.

```bash
python scamshield.py                 # show metrics, then check messages interactively
python scamshield.py "your message"  # check one message
python -m unittest test_scamshield.py
```

## Project layout

```
scamshield.py        # tokenizer, Naive Bayes model, evaluation, CLI
data/messages.csv    # labelled training data (text,label)
test_scamshield.py   # unit tests
```

## Limitations and next steps

- The small synthetic dataset means it can be fooled by unusual wording; it is not a replacement for caution.
- Ideas: train on a larger real dataset, add character n-grams to catch obfuscation such as "acc0unt", or wrap it in a browser extension or messaging bot.

## License

MIT
