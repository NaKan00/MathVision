import json
from pathlib import Path
from typing import Any, Dict

from im2latex.data.tokenizer import Tokenizer, Vocab, TOKENIZER_VERSION


def save_tokenizer(tok: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "tokenizer_version": TOKENIZER_VERSION,
        "itos": tok.vocab.itos,
        "pad": tok.vocab.pad,
        "bos": tok.vocab.bos,
        "eos": tok.vocab.eos,
        "unk": tok.vocab.unk,
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    itos = data["itos"]
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    vocab = Vocab(
        stoi=stoi,
        itos=itos,
        pad=data["pad"],
        bos=data["bos"],
        eos=data["eos"],
        unk=data["unk"],
    )

    return Tokenizer(vocab)