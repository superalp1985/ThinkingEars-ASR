import gzip
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path


def _word_set(data):
    if isinstance(data, dict):
        data = data.keys()
    elif not isinstance(data, list):
        raise ValueError("Lexicon JSON must be an object or a list of words.")
    if any(not isinstance(word, str) for word in data):
        raise ValueError("Lexicon entries must be strings.")
    return frozenset(word for word in data if word)


@lru_cache(maxsize=1)
def builtin_lexicon():
    resource = files("thinkingears_asr").joinpath("data/lexicon.json.gz")
    with resource.open("rb") as compressed:
        with gzip.open(compressed, mode="rt", encoding="utf-8") as stream:
            return _word_set(json.load(stream))


def load_lexicon(path):
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    name = path.name[:-3] if path.suffix.lower() == ".gz" else path.name
    with opener(path, mode="rt", encoding="utf-8-sig") as stream:
        if name.lower().endswith(".json"):
            return _word_set(json.load(stream))
        return frozenset(line.strip() for line in stream if line.strip())
