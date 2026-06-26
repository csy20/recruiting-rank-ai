import numpy as np

from rank import _encode_with_chunking


class FakeTokenizer:
    model_max_length = 6

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=True,
    ):
        tokens = [int(part.removeprefix("tok")) for part in text.split() if part.startswith("tok")]
        if add_special_tokens:
            tokens = [-1, *tokens, -2]
        if len(tokens) > self.model_max_length and verbose:
            raise AssertionError("long text tokenized with warnings enabled")
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(f"tok{token}" for token in tokens if token >= 0)


class FakeModel:
    max_seq_length = 6

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.encoded_texts = []

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True, **kwargs):
        self.encoded_texts.extend(texts)
        rows = []
        for text in texts:
            tokens = self.tokenizer.encode(text, verbose=False)
            assert len(tokens) <= self.max_seq_length
            value = float(sum(token for token in tokens if token >= 0))
            rows.append(np.array([value, 1.0], dtype=np.float32))
        return np.array(rows)


def test_encode_with_chunking_splits_long_texts_before_model_encode():
    model = FakeModel()

    embs = _encode_with_chunking(
        ["tok1 tok2", "tok1 tok2 tok3 tok4 tok5 tok6 tok7 tok8 tok9"],
        model,
        normalize=False,
        batch_size=2,
    )

    assert embs.shape == (2, 2)
    assert model.encoded_texts == [
        "tok1 tok2",
        "tok1 tok2 tok3 tok4",
        "tok5 tok6 tok7 tok8",
        "tok9",
    ]
    np.testing.assert_allclose(embs[0], np.array([3.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(embs[1], np.array([15.0, 1.0], dtype=np.float32))
