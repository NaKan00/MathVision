import re


def remove_unknown_tokens(s: str) -> str:
    return s.replace("<unk>", "")


def cleanup_spaces(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)

    s = s.replace(r"\,", r"\,")
    s = s.replace(r"\;", r"\;")
    s = s.replace(r"\:", r"\:")

    return s


def cleanup_repeated_spacing(s: str) -> str:
    s = re.sub(r"(\\quad){2,}", r"\\quad", s)
    s = re.sub(r"(\\,){2,}", r"\\,", s)
    s = re.sub(r"(\\;){2,}", r"\\;", s)
    return s


def balance_braces(s: str) -> str:
    out = []
    balance = 0

    for ch in s:
        if ch == "{":
            balance += 1
            out.append(ch)
        elif ch == "}":
            if balance > 0:
                balance -= 1
                out.append(ch)
            else:
                continue
        else:
            out.append(ch)

    if balance > 0:
        out.append("}" * balance)

    return "".join(out)


def fix_left_right(s: str) -> str:
    left_count = s.count(r"\left")
    right_count = s.count(r"\right")

    if left_count == right_count:
        return s

    # Если пары сломаны, лучше убрать \left/\right,
    # чем оставить нерендерящийся LaTeX.
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")

    return s


def cleanup_empty_groups(s: str) -> str:
    s = s.replace("{}", "")
    return s


def normalize_wrappers(s: str) -> str:
    # Снимаем совсем внешние одиночные скобки вида {...},
    # но только если они балансные.
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        inner = s[1:-1]
        if inner.count("{") == inner.count("}"):
            return inner
    return s


def postprocess_latex(s: str) -> str:
    s = remove_unknown_tokens(s)
    s = cleanup_spaces(s)
    s = cleanup_repeated_spacing(s)
    s = fix_left_right(s)
    s = balance_braces(s)
    s = cleanup_empty_groups(s)
    s = normalize_wrappers(s)
    return s