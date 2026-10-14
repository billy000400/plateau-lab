"""One three-word continuation convention for local and hosted results."""
import re

MAX_GENERATION_TOKENS = 48
WORD = re.compile(r"\b[^\W_]+(?:['’\-][^\W_]+)*\b", re.UNICODE)


def next_word_spans(prefix, continuation):
    return [match for match in WORD.finditer(prefix + continuation) if match.start() >= len(prefix)]


def word_prediction(tokenizer, prompt_ids, tokens, probabilities, complete=False):
    """Keep three complete new words plus the look-ahead needed to find their end.

    A suffix completing the prompt's final word isn't a new word. Stop at raw
    greedy EOS; nnsight may have computed later tokens to finish its trace.
    """
    prefix = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)
    generated, kept_probs, matches, text = [], [], [], ""
    for token, probability in zip(tokens[:MAX_GENERATION_TOKENS], probabilities):
        if token == tokenizer.eos_token_id:
            complete = True
            break
        generated.append(token)
        kept_probs.append(probability)
        text = tokenizer.decode(generated, clean_up_tokenization_spaces=False)
        matches = next_word_spans(prefix, text)
        if len(matches) >= 4:
            complete = True
            break
    end = matches[2].end() - len(prefix) if len(matches) >= 3 else len(text)
    return {"continuation": text[:end], "words": [m.group() for m in matches[:3]],
            "word_count": min(3, len(matches)), "complete": complete,
            "tokens": [{"id": token, "text": tokenizer.decode([token]), "probability": p}
                       for token, p in zip(generated, kept_probs)],
            "note": "Token details include look-ahead tokens used to confirm word boundaries. Only the first three new words are displayed."}
