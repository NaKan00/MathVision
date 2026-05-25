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
    s = re.sub(r"~+", "", s)
    s = re.sub(r"\\hspace\{[^{}]*\}", "", s)

    for cmd in SPACE_COMMANDS:
        s = s.replace(cmd, "")

    return s


def normalize_left_right(s: str) -> str:
    wrappers = [
        r"\left",
        r"\right",
        r"\Bigg",
        r"\bigg",
        r"\Big",
        r"\big",
        r"\lBig",
        r"\rBig",
    ]

    for w in wrappers:
        s = s.replace(w, "")

    return s


def normalize_angle_brackets(s: str) -> str:
    s = s.replace(r"\langle", "<")
    s = s.replace(r"\rangle", ">")
    s = s.replace(r"\left<", "<")
    s = s.replace(r"\right>", ">")
    return s


def normalize_operatorname_mathrm(s: str) -> str:
    s = re.sub(r"\\operatorname\{([^{}]+)\}", r"\\mathrm{\1}", s)
    return s


def normalize_frac_shortcuts(s: str) -> str:
    s = re.sub(r"\\frac([0-9])([0-9])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac([A-Za-z])([0-9])", r"\\frac{\1}{\2}", s)
    return s


def normalize_common_latex(s: str) -> str:
    replacements = {
        r"\leqslant": r"\leq",
        r"\geqslant": r"\geq",
        r"\ne": r"\neq",
        r"\to": r"\rightarrow",

        r"\calL": r"\mathcal{L}",
        r"\cal{L}": r"\mathcal{L}",
        r"{\calL}": r"\mathcal{L}",
        r"{\cal{L}}": r"\mathcal{L}",

        r"\bfV": r"\mathbf{V}",
        r"{\bfV}": r"\mathbf{V}",
        r"\bf0": r"\mathbf{0}",
        r"{\bf0}": r"\mathbf{0}",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    return s


def remove_outer_braces_once(s: str) -> str:
    if len(s) < 2 or s[0] != "{" or s[-1] != "}":
        return s

    bal = 0

    for i, ch in enumerate(s):
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1

        if bal == 0 and i != len(s) - 1:
            return s

    return s[1:-1]


def remove_redundant_outer_braces(s: str) -> str:
    for _ in range(5):
        prev = s
        s = remove_outer_braces_once(s)

        if s == prev:
            break

    return s


def normalize_command_wrappers(s: str) -> str:
    s = re.sub(r"\{(\\[A-Za-z]+)\}\^", r"\1^", s)
    s = re.sub(r"\{(\\[A-Za-z]+)\}_", r"\1_", s)

    s = re.sub(r"\{([A-Za-z0-9])\}\^", r"\1^", s)
    s = re.sub(r"\{([A-Za-z0-9])\}_", r"\1_", s)

    return s


def normalize_bar_hat_groups(s: str) -> str:
    s = re.sub(r"\{\\bar\{([^{}]+)\}\}", r"\\bar{\1}", s)
    s = re.sub(r"\{\\hat\{([^{}]+)\}\}", r"\\hat{\1}", s)
    s = re.sub(r"\{\\tilde\{([^{}]+)\}\}", r"\\tilde{\1}", s)
    s = re.sub(r"\{\\widetilde\{([^{}]+)\}\}", r"\\widetilde{\1}", s)
    s = re.sub(r"\{\\overline\{([^{}]+)\}\}", r"\\overline{\1}", s)
    return s


def normalize_frac_groups(s: str) -> str:
    s = re.sub(r"\{\\frac\{", r"\\frac{", s)
    return s


def normalize_wrapped_functions(s: str) -> str:
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
        "overline",
        "not",
        "partial",
        "delta",
        "alpha",
        "beta",
        "gamma",
        "pi",
        "mu",
        "nu",
        "sigma",
        "theta",
        "lambda",
        "epsilon",
    ]

    for cmd in commands:
        s = re.sub(r"\{(\\" + cmd + r")", r"\1", s)

    s = re.sub(r"\}([\^_=+\-\*/,\)\]\|;:\.])", r"\1", s)

    return s


def normalize_punctuation(s: str) -> str:
    s = s.rstrip(".")
    s = s.rstrip(",")
    return s


def canonicalize_latex(s: str) -> str:
    s = str(s)

    s = strip_spaces(s)
    s = remove_latex_spacing(s)
    s = normalize_left_right(s)
    s = normalize_angle_brackets(s)
    s = normalize_operatorname_mathrm(s)
    s = normalize_frac_shortcuts(s)
    s = normalize_common_latex(s)
    s = normalize_punctuation(s)

    for _ in range(5):
        prev = s

        s = remove_redundant_outer_braces(s)
        s = normalize_bar_hat_groups(s)
        s = normalize_frac_groups(s)
        s = normalize_command_wrappers(s)
        s = normalize_wrapped_functions(s)
        s = normalize_operatorname_mathrm(s)
        s = normalize_frac_shortcuts(s)
        s = normalize_common_latex(s)
        s = normalize_punctuation(s)

        if s == prev:
            break

    return s