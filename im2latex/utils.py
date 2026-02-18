import json
from pathlib import Path
from typing import Any, Dict

from im2latex.data.tokenizer import Tokenizer, Vocab


def save_tokenizer(tok: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "itos": tok.vocab.itos,
        "pad": int(tok.vocab.pad),
        "bos": int(tok.vocab.bos),
        "eos": int(tok.vocab.eos),
        "unk": int(tok.vocab.unk),
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    itos = data["itos"]
    stoi = {token: idx for idx, token in enumerate(itos)}

    vocab = Vocab(
        stoi=stoi,
        itos=itos,
        pad=int(data["pad"]),
        bos=int(data["bos"]),
        eos=int(data["eos"]),
        unk=int(data["unk"]),
    )

    return Tokenizer(vocab)