import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


TOKENIZER_VERSION = "latex_regex_v3"


LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+\*?")
ESCAPED_CHAR_RE = re.compile(r"\\[^A-Za-z\s]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

LEFT_RIGHT_DELIMS = {
    "(",
    ")",
    "[",
    "]",
    ".",
    "|",
    r"\{",
    r"\}",
    r"\langle",
    r"\rangle",
    r"\lfloor",
    r"\rfloor",
    r"\lceil",
    r"\rceil",
    r"\vert",
    r"\Vert",
}


def normalize_latex(s: str) -> str:
    s = str(s)

    s = s.replace("\n", " ")
    s = s.replace("\t", " ")
    s = " ".join(s.split())

    s = s.replace("\\ ", " ")
    s = s.replace(" {", "{")
    s = s.replace("{ ", "{")
    s = s.replace(" }", "}")
    s = s.replace("_ ", "_")
    s = s.replace("^ ", "^")

    return s.strip()


def _read_left_right_delim(s: str, i: int) -> tuple[str | None, int]:
    if i >= len(s):
        return None, i

   
    if s[i] == "\\":
        m = LATEX_CMD_RE.match(s, i)
        if m:
            delim = m.group()
            if delim in LEFT_RIGHT_DELIMS:
                return delim, m.end()

        m = ESCAPED_CHAR_RE.match(s, i)
        if m:
            delim = m.group()
            if delim in LEFT_RIGHT_DELIMS:
                return delim, m.end()

   
    delim = s[i]
    if delim in LEFT_RIGHT_DELIMS:
        return delim, i + 1

    return None, i


def latex_tokenize(s: str) -> List[str]:
    s = normalize_latex(s)

    tokens: List[str] = []
    i = 0

    while i < len(s):
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        
        if s.startswith(r"\left", i):
            j = i + len(r"\left")
            delim, end = _read_left_right_delim(s, j)
            if delim is not None:
                tokens.append(r"\left" + delim)
                i = end
                continue

        if s.startswith(r"\right", i):
            j = i + len(r"\right")
            delim, end = _read_left_right_delim(s, j)
            if delim is not None:
                tokens.append(r"\right" + delim)
                i = end
                continue

       
        m = LATEX_CMD_RE.match(s, i)
        if m:
            tokens.append(m.group())
            i = m.end()
            continue

       
        m = ESCAPED_CHAR_RE.match(s, i)
        if m:
            tokens.append(m.group())
            i = m.end()
            continue

        
        m = NUMBER_RE.match(s, i)
        if m:
            tokens.append(m.group())
            i = m.end()
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
        self.version = TOKENIZER_VERSION

    @staticmethod
    def build(
        formulas: List[str],
        min_freq: int = 2,
        max_size: int = 8000,
    ) -> "Tokenizer":
        counter = Counter()

        for formula in formulas:
            counter.update(latex_tokenize(formula))

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

        stoi = {tok: idx for idx, tok in enumerate(itos)}

        vocab = Vocab(
            stoi=stoi,
            itos=itos,
            pad=stoi["<pad>"],
            bos=stoi["<bos>"],
            eos=stoi["<eos>"],
            unk=stoi["<unk>"],
        )

        return Tokenizer(vocab)

    def encode(
        self,
        formula: str,
        add_special: bool = True,
    ) -> List[int]:
        tokens = latex_tokenize(formula)

        ids = [
            self.vocab.stoi.get(tok, self.vocab.unk)
            for tok in tokens
        ]

        if add_special:
            ids = [self.vocab.bos] + ids + [self.vocab.eos]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        special_ids = {
            self.vocab.pad,
            self.vocab.bos,
            self.vocab.eos,
        }

        tokens: List[str] = []

        for idx in ids:
            if skip_special and idx in special_ids:
                continue

            if 0 <= idx < len(self.vocab.itos):
                tok = self.vocab.itos[idx]
            else:
                tok = "<unk>"

            if skip_special and tok == "<unk>":
                continue

            tokens.append(tok)

        return "".join(tokens)

    def token_to_id(self, token: str) -> int:
        return self.vocab.stoi.get(token, self.vocab.unk)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.vocab.itos):
            return self.vocab.itos[idx]
        return "<unk>"

    def vocab_size(self) -> int:
        return len(self.vocab.itos)