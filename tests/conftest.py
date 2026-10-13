import random

import pytest

from pyteg import CooccurrenceIndex


def fruit_corpus():
    """A tiny corpus where 'fruit' is general and 'strawberry' specific."""
    docs = []
    for i in range(40):
        doc = ["i", "ate", "some", "fruit", "today"]
        if i < 10:
            doc.append("strawberry")
        elif i < 20:
            doc.append("banana")
        docs.append(doc)
    for i in range(60):
        doc = ["the", "weather", "was", "nice", "today"]
        if i % 6 == 0:
            doc.append("fruit")
        docs.append(doc)
    return docs


@pytest.fixture
def fruit_index():
    return CooccurrenceIndex.build(fruit_corpus())


@pytest.fixture
def random_docs():
    rng = random.Random(13)
    vocab = [f"w{i}" for i in range(40)]
    weights = [1.0 / (i + 1) for i in range(40)]
    return [rng.choices(vocab, weights=weights, k=rng.randint(1, 15)) for _ in range(300)]
