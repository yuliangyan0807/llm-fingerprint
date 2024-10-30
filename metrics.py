from evaluate import load
import numpy as np

# standard edit distance
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # deletion
                           dp[i][j - 1] + 1,  # insertion
                           dp[i - 1][j - 1] + cost)  # substitution

    return dp[m][n]

# weithted edit distance
def weighted_edit_distance(s1, s2, conf1, conf2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + (1 - conf1[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + (1 - conf2[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = abs(conf1[i - 1] - conf2[j - 1]) if s1[i - 1] != s2[j - 1] else 0
            dp[i][j] = min(dp[i - 1][j] + (1 - conf1[i - 1]),  # deletion
                           dp[i][j - 1] + (1 - conf2[j - 1]),  # insertion
                           dp[i - 1][j - 1] + cost)            # substitution

    return dp[m][n]

def jaccard_similarity(tokens1, tokens2):
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union != 0 else 0

def calculate_varentropy(logits):
    """
    Calculate the Varentropy of the given logits.

    Parameters:
    - logits: A numpy array of logits (shape: [num_tokens, vocab_size])

    Returns:
    - varentropy: The computed Varentropy value
    """
    # Convert logits to probabilities using softmax
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Calculate the entropy for each token
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)  # Adding a small value to avoid log(0)

    # Calculate the variance of the entropies
    varentropy = np.var(entropy)

    return varentropy

# semantic similarity
def get_semantic_similarity_with_bert_score(text_0, text_1):
    """
    Given two text generated by two LLMs, compute their semantic similarity with bert score.
    """
    bert_score = load("bertscore")
    predictions = [text_1]
    references = [text_0]
    
    results = bert_score.compute(predictions=predictions,
                                 references=references,
                                 lang='en')
    
    return results['f1'][0]

# perplexity
def get_perplexity(text):
    """
    Given the generated text, return the perplexity of the text under llama model.
    """
    perplexity = load("perplexity", module_type="metric")
    
    results = perplexity.compute(predictions=[text],
                                 model_id='/mnt/data/yuliangyan/meta-llama/Meta-Llama-3.1-8B/')
    
    return results['mean_perplexity']        