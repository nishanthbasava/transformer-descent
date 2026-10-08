import json
from pathlib import Path #lets me work with filesystem paths as objects instead of strings

# the vocab built from the sorted set of characters in input.txt, saved to vocab.jcon so the C++ stages can read it later
ROOT = Path(__file__).resolve().parents[2]
text = (ROOT / "data" / "input.txt").read_text()

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)} # string to integer, 'a' -> 39
itos = {i: c for c, i in stoi.items()} # integer to string, 39 -> 'a'

#encode (text) -> list(int) 
def encode(s):
    return [stoi[c] for c in s]

# decode(ids) -> str
def decode (ids):
    return "".join(itos[i] for i in ids)

if __name__ == "__main__":
    (Path(__file__).parent / "vocab.json").write_text(json.dumps(chars))
    assert decode(encode("hello")) == "hello"

# a fixed split: first 90% of the data is train, last 10% is val, used by every stage
data = encode(text)
n = int(0.9 * len(data))
train_ids = data[:n]
val_ids = data[n:]

# BPE later