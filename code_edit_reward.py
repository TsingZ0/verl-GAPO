def compute_score_em(data_source, solution_str, ground_truth, extra_info=None):
    res = {}
    res['exact_match'] = exact_match(solution_str, ground_truth)
    res['edit_distance'] = edit_distance(solution_str, ground_truth)
    res['bleu'] = bleu_score(solution_str, ground_truth)
    res['score'] = res['exact_match']  # Use exact match as the reward
    return res

def compute_score_ed(data_source, solution_str, ground_truth, extra_info=None):
    res = {}
    res['exact_match'] = exact_match(solution_str, ground_truth)
    res['edit_distance'] = edit_distance(solution_str, ground_truth)
    res['bleu'] = bleu_score(solution_str, ground_truth)
    res['score'] = res['edit_distance']  # Use edit distance as the reward
    return res

def compute_score_edem(data_source, solution_str, ground_truth, extra_info=None):
    res = {}
    res['exact_match'] = exact_match(solution_str, ground_truth)
    res['edit_distance'] = edit_distance(solution_str, ground_truth)
    res['bleu'] = bleu_score(solution_str, ground_truth)
    res['score'] = (res['edit_distance'] + res['exact_match']) / 2  # Use edit distance + exact match as the reward
    return res

def compute_score_bleu(data_source, solution_str, ground_truth, extra_info=None):
    res = {}
    res['exact_match'] = exact_match(solution_str, ground_truth)
    res['edit_distance'] = edit_distance(solution_str, ground_truth)
    res['bleu'] = bleu_score(solution_str, ground_truth)
    res['score'] = res['bleu']  # Use bleu as the reward
    return res


def exact_match(solution_str, ground_truth):
    return int(solution_str.strip() == ground_truth.strip())

def edit_distance(solution_str, ground_truth):
    """
    Computes normalized edit distance reward between solution_str and ground_truth.
    Reward is 1.0 if identical, 0.0 if completely different.
    """
    def levenshtein(a, b):
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[n][m]

    a = solution_str.strip()
    b = ground_truth.strip()
    dist = levenshtein(a, b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len

def bleu_score(solution_str, ground_truth):
    """
    Computes BLEU score reward between solution_str and ground_truth.
    Reward is 1.0 if identical, 0.0 if completely different.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        raise ImportError("Please install nltk to use BLEU score: pip install nltk")

    reference = [ground_truth.strip().split()]
    hypothesis = solution_str.strip().split()
    if not reference[0] or not hypothesis:
        return 0.0
    smoothie = SmoothingFunction().method1
    score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
    return score