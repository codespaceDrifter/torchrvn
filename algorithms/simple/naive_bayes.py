'''
assume all data is a tensor of int ids.  
bag of words, ignore positional information  

conditional probabiliy:  
P(A|B) = P(A ∩ B) / P(B)  
P(A ∩ B) = P(A|B) * P(B)  

bayes rule:  
P(A|B) = P(B|A) * P(A) / P(B)  

law of total probabiliy:
P(A) = P(A|B_1)P(B_1) + ... + P(A|B_n)P(B_n)  
P(B_i | A) = P(A | B_i) P (B_i) / P(A)  

say A is made up of several features a, b, c and we want P(Y | a, b, c)

chain rule (exact):     P(a, b, c | Y) = P(a | Y) * P(b | a, Y) * P(c | a, b, Y)
naive bayes (assumes conditional independence): P(a, b, c | Y) = P(a | Y) * P(b | Y) * P(c | Y)

applying bayes rule:
P(Y | a, b, c) = P(a, b, c | Y) * P(Y) / P(a, b, c)

substituting naive bayes assumption:
P(Y | a, b, c) = P(a | Y) * P(b | Y) * P(c | Y) * P(Y) / P(a, b, c)

P(a, b, c) is constant across all classes, so we ignore it and compare numerators to find argmax Y

laplace smoothing:
if a feature value was never seen with a class, P(x | Y) = 0, which zeros out the entire product
to prevent this, add 1 to every count numerator and vocab size to denominator:

P(x | Y = y) = (count(x, y) + 1) / (count(y) + |V|)

so P(x | Y = y) still sums to 1 over all x

underflow prevention:
multiplying many small probabilities causes underflow, so use log: log(xy) = log(x) + log(y)
log P(Y | features) = log P(Y) + Σ log P( count (x_i) + 1 / Y + |V| )

gaussian naive bayes (continuous features):
for each class y and feature i, model P(x_i | y) as Gaussian with mean μ and variance σ²

P(x_i | y) = (1 / sqrt(2πσ²)) * exp(-(x_i - μ)² / (2σ²))

in log space:
log P(x_i | y) = -0.5 * log(2π) - 0.5 * log(σ²) - (x_i - μ)² / (2σ²)

for classification:
log P(y | x) ∝ log P(y) + Σ_i log P(x_i | y)

variance smoothing:
σ²_smoothed = σ² + ε * max(σ²) (prevents zero variance, scale-invariant)

density is not probability, but argmax is the same because dx terms cancel when comparing:
P(x₁ ∈ dx₁, x₂ ∈ dx₂, ... | y) ≈ f(x₁|y) * f(x₂|y) * ... * dx₁ * dx₂ * ...
'''

import torch
from collections import defaultdict


class NaiveBayes:
    """
    True naive bayes for next token prediction.
    Bag of words - ignores position, assumes context tokens are independent.
    P(next | context) ∝ P(next) * Π P(context_token | next)
    """

    def __init__(self, vocab_size: int, n: int = 5):
        self.vocab_size = vocab_size
        # n = context window size (bag of n tokens)
        self.n = n
        # raw counts during training
        self._likelihood_counts = defaultdict(lambda: defaultdict(int))
        self._prior_counts = defaultdict(int)
        self._total = 0
        # precomputed after finalize()
        # (vocab_size,) log P(next)
        self.log_prior = None
        # (vocab_size,) baseline log prob per next_tok for unseen ctx
        self.baseline = None
        # {ctx_tok: {next_tok: log_prob}} - sparse, only seen pairs
        self.log_likelihood = None

    def train(self, tokens: torch.Tensor):
        """
        tokens: (seq_len,) int tensor
        for each position, treat previous n tokens as bag of words
        """
        tokens = tokens.flatten().tolist()
        for i in range(self.n, len(tokens)):
            context = tokens[i - self.n:i]
            target = tokens[i]
            self._prior_counts[target] += 1
            self._total += 1
            for tok in context:
                self._likelihood_counts[target][tok] += 1

    def finalize(self):
        """
        precompute log probabilities from counts. call once after all training.
        """
        # (vocab_size,) log P(next) with laplace
        self.log_prior = torch.zeros(self.vocab_size)
        for next_tok in range(self.vocab_size):
            count = self._prior_counts[next_tok]
            self.log_prior[next_tok] = torch.log(torch.tensor((count + 1) / (self._total + self.vocab_size)))

        # (vocab_size,) baseline log prob for unseen context tokens
        self.baseline = torch.zeros(self.vocab_size)
        # {ctx_tok: {next_tok: log_prob}} sparse storage
        self.log_likelihood = defaultdict(dict)

        for next_tok in range(self.vocab_size):
            total_ctx = sum(self._likelihood_counts[next_tok].values())
            denom = total_ctx + self.vocab_size
            # baseline for this next_tok
            self.baseline[next_tok] = torch.log(torch.tensor(1.0 / denom))
            # only store seen (ctx_tok, next_tok) pairs
            for ctx_tok, count in self._likelihood_counts[next_tok].items():
                log_prob = torch.log(torch.tensor((count + 1) / denom))
                self.log_likelihood[ctx_tok][next_tok] = log_prob

        # free raw counts
        self._likelihood_counts = None
        self._prior_counts = None

    def get_log_probs(self, context: list[int]) -> torch.Tensor:
        """
        context: list of n ints (treated as bag)
        returns (vocab_size,) unnormalized log scores
        """
        assert self.log_prior is not None, "call finalize() first"
        # start with prior + baseline for each context token
        # (vocab_size,)
        scores = self.log_prior + self.baseline * len(context)

        # add delta (actual - baseline) for seen pairs only
        for ctx_tok in context:
            if ctx_tok in self.log_likelihood:
                for next_tok, log_prob in self.log_likelihood[ctx_tok].items():
                    scores[next_tok] += (log_prob - self.baseline[next_tok])

        return scores

    def predict(self, context: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        context: (n,) tensor of last n tokens (treated as bag)
        returns (predicted_token, log_probs of shape (vocab_size,))
        """
        if self.log_prior is None:
            self.finalize()
        # (vocab_size,)
        log_probs = self.get_log_probs(context.tolist())
        pred = log_probs.argmax().item()
        return pred, log_probs

    def generate(self, context: torch.Tensor, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        context: (n,) initial context
        returns (generated_tokens, all_log_probs)
        generated_tokens: (max_tokens,)
        all_log_probs: (max_tokens, vocab_size)
        """
        ctx = context.clone()
        generated = []
        all_log_probs = []

        for _ in range(max_tokens):
            pred, log_probs = self.predict(ctx)
            generated.append(pred)
            all_log_probs.append(log_probs)
            # shift context: drop first, append pred -> (n,)
            ctx = torch.cat([ctx[1:], torch.tensor([pred])])

        # (max_tokens,)
        generated = torch.tensor(generated)
        # (max_tokens, vocab_size)
        all_log_probs = torch.stack(all_log_probs)
        return generated, all_log_probs


class GaussianNaiveBayes:
    """
    Gaussian naive bayes for classification with continuous features.
    For each class y and feature i, models P(x_i | y) as Gaussian.
    P(y | x) ∝ P(y) * Π_i P(x_i | y)
    """

    def __init__(self, var_smoothing: float = 1e-9):
        # ε added to all variances to prevent zero variance
        self.var_smoothing = var_smoothing
        # computed after train()
        # (n_classes,) log P(y)
        self.log_prior = None
        # (n_classes, n_features) mean per class per feature
        self.class_means = None
        # (n_classes, n_features) variance per class per feature (smoothed)
        self.class_vars = None
        # e.g. [3, 7, 42] - unique class labels sorted, maps index i -> class label
        self.classes = None

    def train(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n_samples, n_features) continuous features
        y: (n_samples,) class labels (ints)
        """
        # unique classes sorted
        self.classes = torch.unique(y).tolist()
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # (n_classes, n_features)
        self.class_means = torch.zeros(n_classes, n_features)
        self.class_vars = torch.zeros(n_classes, n_features)
        # (n_classes,)
        self.log_prior = torch.zeros(n_classes)

        for i, c in enumerate(self.classes):
            # (n_samples_c, n_features)
            X_c = X[y == c]
            self.class_means[i] = X_c.mean(dim=0)
            self.class_vars[i] = X_c.var(dim=0, unbiased=False)
            self.log_prior[i] = torch.log(torch.tensor(len(X_c) / len(X)))

        # variance smoothing: add ε * max_var to all variances
        self.class_vars += self.var_smoothing * self.class_vars.max()

    def get_log_probs(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (n_samples, n_features)
        returns: (n_samples, n_classes) log P(y | x) unnormalized
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        # (n_samples, n_classes)
        log_probs = torch.zeros(n_samples, n_classes)

        for i in range(n_classes):
            # (n_features,)
            mu = self.class_means[i]
            sigma2 = self.class_vars[i]
            # (n_samples, n_features) -> (n_samples,)
            # log P(x | y) = Σ_i [-0.5 * log(2π) - 0.5 * log(σ²) - (x - μ)² / (2σ²)]
            log_likelihood = -0.5 * (torch.log(2 * torch.pi * sigma2) + (X - mu) ** 2 / sigma2)
            # (n_samples, n_features) -> (n_samples,). summing over all the input features. 
            log_probs[:, i] = log_likelihood.sum(dim=1) + self.log_prior[i]

        return log_probs

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        X: (n_samples, n_features)
        returns: (predictions, log_probs)
            predictions: (n_samples,) predicted class labels
            log_probs: (n_samples, n_classes)
        """
        assert self.class_means is not None, "call train() first"
        # (n_samples, n_classes)
        log_probs = self.get_log_probs(X)
        # (n_samples,) indices into self.classes
        pred_indices = log_probs.argmax(dim=1)
        # map back to actual class labels
        predictions = torch.tensor([self.classes[i] for i in pred_indices])
        return predictions, log_probs
