import re


SPACE_COMMANDS = [
    r"\quad",
    r"\qquad",
    r"\,",
    r"\;",
    r"\:",
    r"\!",
]


def strip_spaces(s: str) -> str:
    return "".join(str(s).split())


def remove_latex_spacing(s: str) -> str:
    for cmd in SPACE_COMMANDS:
        s = s.replace(cmd, "")
    return s


def remove_outer_braces_once(s: str) -> str:
    if len(s) < 2:
        return s

    if not (s[0] == "{" and s[-1] == "}"):
        return s

    balance = 0

    for i, ch in enumerate(s):
        if ch == "{":
            balance += 1
        elif ch == "}":
            balance -= 1

        if balance == 0 and i != len(s) - 1:
            return s

    return s[1:-1]


def remove_redundant_outer_braces(s: str) -> str:
    prev = None

    while prev != s:
        prev = s
        s = remove_outer_braces_once(s)

    return s


def normalize_grouped_commands(s: str) -> str:
    # {\frac{...}{...}} -> \frac{...}{...}
    commands = [
        "frac",
        "sqrt",
        "bar",
        "tilde",
        "widetilde",
        "hat",
        "widehat",
        "vec",
        "dot",
        "ddot",
        "mathrm",
        "mathbf",
        "mathit",
        "mathcal",
        "operatorname",
    ]

    for cmd in commands:
        pattern = r"\{(\\" + cmd + r")"
        s = re.sub(pattern, r"\1", s)

    # remove simple command-closing redundant brace before ^/_/= etc.
    s = re.sub(r"\}([\^_=+\-\),\]\}])", r"\1", s)

    return s


def normalize_single_symbol_powers(s: str) -> str:
    # {\pi}^{2} -> \pi^{2}
    s = re.sub(r"\{(\\[A-Za-z]+)\}\^", r"\1^", s)

    # {x}^{2} -> x^{2}
    s = re.sub(r"\{([A-Za-z0-9])\}\^", r"\1^", s)

    # {\delta}_{ab} -> \delta_{ab}
    s = re.sub(r"\{(\\[A-Za-z]+)\}_", r"\1_", s)

    # {x}_{i} -> x_{i}
    s = re.sub(r"\{([A-Za-z0-9])\}_", r"\1_", s)

    return s


def normalize_left_right(s: str) -> str:
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    return s


def normalize_common_latex(s: str) -> str:
    replacements = {
        r"\leqslant": r"\leq",
        r"\geqslant": r"\geq",
        r"\ne": r"\neq",
        r"\to": r"\rightarrow",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    return s


def canonicalize_latex(s: str) -> str:
    s = str(s)

    s = strip_spaces(s)
    s = remove_latex_spacing(s)
    s = normalize_left_right(s)
    s = normalize_common_latex(s)

    prev = None
    while prev != s:
        prev = s
        s = remove_redundant_outer_braces(s)
        s = normalize_single_symbol_powers(s)
        s = normalize_grouped_commands(s)

    return s