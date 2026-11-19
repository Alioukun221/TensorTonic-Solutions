def wordpiece_tokenize(
    text: str,
    vocab: dict,
    unk_token: str = "[UNK]",
    max_word_length: int = 100,
) -> list[str]:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    tokens: list[str] = []

    for word in text.lower().split():
        word_length = len(word)

        if word_length > max_word_length:
            tokens.append(unk_token)
            continue

        start = 0
        sub_tokens: list[str] = []
        is_bad = False

        while start < word_length:
            end = word_length
            current_substr = None

            while start < end:
                substr = word[start:end] if start == 0 else "##" + word[start:end]
                if substr in vocab:
                    current_substr = substr
                    break
                end -= 1

            if current_substr is None:
                is_bad = True
                break

            sub_tokens.append(current_substr)
            start = end

        if is_bad:
            tokens.append(unk_token)
        else:
            tokens.extend(sub_tokens)

    return tokens