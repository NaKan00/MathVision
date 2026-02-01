import json
from pathlib import Path
from typing import Any, Dict

from im2latex.data.tokenizer import Tokenizer, Vocab


def save_tokenizer(tok: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "itos": tok.vocab.itos,
        "pad": tok.vocab.pad,
        "bos": tok.vocab.bos,
        "eos": tok.vocab.eos,
        "unk": tok.vocab.unk,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    itos = data["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    vocab = Vocab(
        stoi=stoi,
        itos=itos,
        pad=data["pad"],
        bos=data["bos"],
        eos=data["eos"],
        unk=data["unk"],
    )
    return Tokenizer(vocab)