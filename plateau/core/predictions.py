"""Small, portable prediction records shared by local and hosted inference."""
import math


def matrix_enabled(request):
    enabled = request.get("sample_token_matrix", False)
    if not isinstance(enabled, bool):
        raise ValueError("sample_token_matrix must be true or false.")
    return enabled


def token_prediction(tokenizer, t, token):
    token = int(token)
    return {"t": t, "token_id": token, "token": tokenizer.decode([token])}


def sample_prediction(tokenizer, t, ids, probabilities):
    """Format steps × ranks; later steps condition on earlier rank-one tokens.

    Forwards after a greedy EOS are discarded, even if other batch rows needed
    them. Nonfinite distributions are unavailable rather than made-up scores.
    """
    first = int(ids[0][0])
    steps, stop_reason = [], None
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    for tokens, probs in zip(ids, probabilities):
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs):
            stop_reason = "nonfinite"
            break
        steps.append([{"id": int(token), "text": tokenizer.decode([int(token)]),
                       "probability": float(p), "is_eos": int(token) in eos_ids}
                      for token, p in zip(tokens, probs)])
        if steps[-1][0]["is_eos"]:
            stop_reason = "eos"
            break
    return {**token_prediction(tokenizer, t, first),
            "token_matrix": {"steps": steps, "stop_reason": stop_reason}}
