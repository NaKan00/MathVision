import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict

LATEX_CMD = re.compile(r"\\[A-Za-z]+")  


def latex_tokenize(s: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(s):
        m = LATEX_CMD.match(s, i)
        if m:
            tokens.append(m.group())
            i = m.end()
            continue

        ch = s[i]

        
        if ch.isspace():
            i += 1
            continue

        tokens.append(ch)
        i += 1
    return tokens


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    pad: int
    bos: int
    eos: int
    unk: int


class Tokenizer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    @staticmethod
    def build(formulas: List[str], min_freq: int = 2, max_size: int = 8000) -> "Tokenizer":
        counter = Counter()
        for f in formulas:
            counter.update(latex_tokenize(f))

        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        itos = specials[:]

        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok in specials:
                continue
            itos.append(tok)
            if len(itos) >= max_size:
                break

        stoi = {t: i for i, t in enumerate(itos)}
        vocab = Vocab(
            stoi=stoi,
            itos=itos,
            pad=stoi["<pad>"],
            bos=stoi["<bos>"],
            eos=stoi["<eos>"],
            unk=stoi["<unk>"],
        )
        return Tokenizer(vocab)

    def encode(self, formula: str, add_special: bool = True) -> List[int]:
        toks = latex_tokenize(formula)
        ids = [self.vocab.stoi.get(t, self.vocab.unk) for t in toks]
        if add_special:
            ids = [self.vocab.bos] + ids + [self.vocab.eos]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            if skip_special and i in (self.vocab.pad, self.vocab.bos, self.vocab.eos):
                continue
            toks.append(self.vocab.itos[i] if 0 <= i < len(self.vocab.itos) else "<unk>")
        
        return "".join(toks)