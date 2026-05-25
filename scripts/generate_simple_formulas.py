import csv
import random
from pathlib import Path


OUT = Path("datasets/simple_formulas/simple_formulas.csv")
N = 50000

VARS = ["x", "y", "z", "a", "b", "c", "m", "n", "t", "u", "v"]
GREEK = ["\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\mu"]


CATEGORY_COUNTS = {
    "algebra": 9000,
    "fractions": 6000,
    "sqrt": 4500,
    "trig": 4500,
    "derivatives": 5000,
    "integrals": 5000,
    "limits": 3500,
    "sums": 3500,
    "physics": 4000,
    "probability": 2500,
    "matrices": 1500,
    "systems": 1500,
}


def rv():
    return random.choice(VARS)


def rg():
    return random.choice(GREEK)


def ri(a=1, b=9):
    return random.randint(a, b)


def coeff_var(k: int, x: str) -> str:
    if k == 1:
        return x
    if k == -1:
        return f"-{x}"
    return f"{k}{x}"


def term(k: int, body: str, first: bool = False) -> str:
    if k == 0:
        return ""

    if first:
        if k == 1:
            return body
        if k == -1:
            return f"-{body}"
        return f"{k}{body}"

    if k > 0:
        if k == 1:
            return f"+{body}"
        return f"+{k}{body}"

    if k == -1:
        return f"-{body}"

    return f"{k}{body}"


def clean_expr(s: str) -> str:
    s = s.replace("+-", "-")
    s = s.replace("--", "+")
    s = s.replace("+ -", "-")
    s = s.replace("- -", "+")
    return s


def algebra_formula():
    x, y, z = rv(), rv(), rv()
    kind = random.choice([
        "poly2",
        "poly3",
        "square_plus",
        "square_minus",
        "diff_squares",
        "pythagorean",
        "quadratic",
        "quadratic_solution",
        "vieta_sum",
        "vieta_prod",
    ])

    if kind == "poly2":
        s = (
            term(ri(1, 6), f"{x}^{{2}}", first=True)
            + term(random.choice([-1, 1]) * ri(1, 9), x)
            + term(random.choice([-1, 1]) * ri(1, 9), "")
        )
        return clean_expr(s)

    if kind == "poly3":
        s = (
            term(ri(1, 5), f"{x}^{{3}}", first=True)
            + term(random.choice([-1, 1]) * ri(1, 6), f"{x}^{{2}}")
            + term(random.choice([-1, 1]) * ri(1, 9), x)
            + term(random.choice([-1, 1]) * ri(1, 9), "")
        )
        return clean_expr(s)

    if kind == "square_plus":
        return f"({x}+{y})^{{2}}={x}^{{2}}+2{x}{y}+{y}^{{2}}"

    if kind == "square_minus":
        return f"({x}-{y})^{{2}}={x}^{{2}}-2{x}{y}+{y}^{{2}}"

    if kind == "diff_squares":
        return f"{x}^{{2}}-{y}^{{2}}=({x}-{y})({x}+{y})"

    if kind == "pythagorean":
        return random.choice([
            f"{x}^{{2}}+{y}^{{2}}={z}^{{2}}",
            f"{z}=\\sqrt{{{x}^{{2}}+{y}^{{2}}}}",
            f"{x}=\\sqrt{{{z}^{{2}}-{y}^{{2}}}}",
        ])

    if kind == "quadratic":
        a, b, c = ri(1, 5), random.choice([-1, 1]) * ri(1, 10), random.choice([-1, 1]) * ri(1, 10)
        return clean_expr(f"{coeff_var(a, x)}^{{2}}{term(b, x)}{term(c, '')}=0")

    if kind == "quadratic_solution":
        return random.choice([
            f"{x}=\\frac{{-b\\pm\\sqrt{{b^{{2}}-4ac}}}}{{2a}}",
            f"D=b^{{2}}-4ac",
            f"{x}_{{1,2}}=\\frac{{-b\\pm\\sqrt{{D}}}}{{2a}}",
        ])

    if kind == "vieta_sum":
        return f"{x}_{{1}}+{x}_{{2}}=-\\frac{{b}}{{a}}"

    return f"{x}_{{1}}{x}_{{2}}=\\frac{{c}}{{a}}"


def fraction_formula():
    x, y, z = rv(), rv(), rv()
    kind = random.choice(["simple", "sum_same_den", "nested", "rational_eq"])

    if kind == "simple":
        num = random.choice([
            f"{coeff_var(ri(1, 9), x)}+{ri()}",
            f"{x}^{{2}}+{ri()}",
            f"{coeff_var(ri(1, 5), x)}-{coeff_var(ri(1, 5), y)}",
            f"{x}+{y}",
        ])
        den = random.choice([
            f"{coeff_var(ri(1, 9), y)}-{ri()}",
            f"{y}^{{2}}+{ri()}",
            f"{x}-{y}",
            f"{ri()}",
        ])
        return f"\\frac{{{num}}}{{{den}}}"

    if kind == "sum_same_den":
        return f"\\frac{{{x}}}{{{y}}}+\\frac{{{z}}}{{{y}}}=\\frac{{{x}+{z}}}{{{y}}}"

    if kind == "nested":
        return f"\\frac{{\\frac{{{x}}}{{{y}}}}}{{{z}}}=\\frac{{{x}}}{{{y}{z}}}"

    return f"\\frac{{{x}+1}}{{{x}-1}}=\\frac{{{y}}}{{{z}}}"


def sqrt_formula():
    x, y = rv(), rv()
    return random.choice([
        f"\\sqrt{{{x}^{{2}}+{y}^{{2}}}}",
        f"\\sqrt{{{ri()}{x}^{{2}}+{ri()}{y}^{{2}}}}",
        f"\\sqrt{{{x}+{ri()}}}",
        f"\\sqrt{{{x}^{{2}}-{y}^{{2}}}}",
        f"\\sqrt[3]{{{x}^{{2}}+{ri()}}}",
    ])


def trig_formula():
    x = rv()
    return random.choice([
        f"\\sin^{{2}}({x})+\\cos^{{2}}({x})=1",
        f"1+\\tan^{{2}}({x})=\\frac{{1}}{{\\cos^{{2}}({x})}}",
        f"\\sin(2{x})=2\\sin({x})\\cos({x})",
        f"\\cos(2{x})=\\cos^{{2}}({x})-\\sin^{{2}}({x})",
        f"\\tan({x})=\\frac{{\\sin({x})}}{{\\cos({x})}}",
    ])


def derivative_formula():
    x = rv()
    p = ri(2, 8)

    return random.choice([
        f"\\frac{{d}}{{d{x}}}{x}^{{{p}}}={p}{x}^{{{p - 1}}}",
        f"\\frac{{d}}{{d{x}}}\\sin({x})=\\cos({x})",
        f"\\frac{{d}}{{d{x}}}\\cos({x})=-\\sin({x})",
        f"\\frac{{d}}{{d{x}}}e^{{{x}}}=e^{{{x}}}",
        f"\\frac{{d}}{{d{x}}}\\ln({x})=\\frac{{1}}{{{x}}}",
        f"f^{{\\prime}}({x})={ri()}{x}+{ri()}",
        f"y^{{\\prime}}={ri()}{x}^{{2}}+{ri()}{x}+{ri()}",
    ])


def integral_formula():
    x = rv()
    p = ri(1, 7)

    return random.choice([
        f"\\int {x}^{{{p}}}\\,d{x}=\\frac{{{x}^{{{p + 1}}}}}{{{p + 1}}}+C",
        f"\\int \\sin({x})\\,d{x}=-\\cos({x})+C",
        f"\\int \\cos({x})\\,d{x}=\\sin({x})+C",
        f"\\int e^{{{x}}}\\,d{x}=e^{{{x}}}+C",
        f"\\int \\frac{{1}}{{{x}}}\\,d{x}=\\ln|{x}|+C",
        f"\\int_{{0}}^{{1}}{x}^{{{p}}}\\,d{x}=\\frac{{1}}{{{p + 1}}}",
    ])


def limit_formula():
    x = rv()

    return random.choice([
        f"\\lim_{{{x}\\to0}}\\frac{{\\sin({x})}}{{{x}}}=1",
        f"\\lim_{{{x}\\to0}}\\frac{{1-\\cos({x})}}{{{x}^{{2}}}}=\\frac{{1}}{{2}}",
        f"\\lim_{{{x}\\to\\infty}}\\frac{{1}}{{{x}}}=0",
        f"\\lim_{{{x}\\to0}}\\frac{{e^{{{x}}}-1}}{{{x}}}=1",
        f"\\lim_{{{x}\\to1}}\\frac{{{x}^{{2}}-1}}{{{x}-1}}=2",
    ])


def sum_formula():
    n = random.choice(["n", "N", "m"])

    return random.choice([
        f"\\sum_{{i=1}}^{{{n}}}i=\\frac{{{n}({n}+1)}}{{2}}",
        f"\\sum_{{i=1}}^{{{n}}}i^{{2}}=\\frac{{{n}({n}+1)(2{n}+1)}}{{6}}",
        f"\\sum_{{k=0}}^{{{n}}}q^{{k}}=\\frac{{1-q^{{{n}+1}}}}{{1-q}}",
        f"\\sum_{{i=1}}^{{{n}}}1={n}",
    ])


def physics_formula():
    x = rv()
    return random.choice([
        "E=mc^{2}",
        "F=ma",
        "p=mv",
        "V=IR",
        "PV=nRT",
        "v=\\frac{s}{t}",
        "T=2\\pi\\sqrt{\\frac{l}{g}}",
        "E=\\frac{mv^{2}}{2}",
        "p=\\rho gh",
        "Q=mc\\Delta T",
        "F=G\\frac{m_{1}m_{2}}{r^{2}}",
        "\\lambda=\\frac{h}{p}",
        "\\omega=2\\pi f",
        "c=\\lambda f",
        f"{x}=x_{0}+vt",
        f"v=v_{0}+at",
    ])


def probability_formula():
    A, B = "A", "B"

    return random.choice([
        f"P({A}\\cap{B})=P({A})P({B})",
        f"P({A}\\cup{B})=P({A})+P({B})-P({A}\\cap{B})",
        f"P({A}|{B})=\\frac{{P({A}\\cap{B})}}{{P({B})}}",
        f"E(X)=\\sum_{{i}}x_i p_i",
        f"D(X)=E(X^{{2}})-E^{{2}}(X)",
    ])


def matrix_formula():
    a, b, c, d = ri(), ri(), ri(), ri()
    return random.choice([
        "\\begin{pmatrix}"
        f"{a}&{b}\\\\"
        f"{c}&{d}"
        "\\end{pmatrix}",
        "\\det\\begin{pmatrix}"
        f"{a}&{b}\\\\"
        f"{c}&{d}"
        "\\end{pmatrix}"
        f"={a * d - b * c}",
    ])


def system_formula():
    x, y = random.sample(VARS[:6], 2)

    return (
        "\\begin{cases}"
        f"{coeff_var(ri(1, 5), x)}+{coeff_var(ri(1, 5), y)}={ri()}\\\\"
        f"{coeff_var(ri(1, 5), x)}-{coeff_var(ri(1, 5), y)}={ri()}"
        "\\end{cases}"
    )


def greek_formula():
    a, b = rg(), rg()

    return random.choice([
        f"{a}+{b}=0",
        f"{a}^{{2}}+{b}^{{2}}=1",
        f"\\frac{{\\partial {a}}}{{\\partial x}}={b}",
        f"{a}_{{n+1}}={a}_{{n}}+{b}",
    ])


GENS = {
    "algebra": algebra_formula,
    "fractions": fraction_formula,
    "sqrt": sqrt_formula,
    "trig": trig_formula,
    "derivatives": derivative_formula,
    "integrals": integral_formula,
    "limits": limit_formula,
    "sums": sum_formula,
    "physics": physics_formula,
    "probability": probability_formula,
    "matrices": matrix_formula,
    "systems": system_formula,
}


def main():
    random.seed(42)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()

    for category, target_count in CATEGORY_COUNTS.items():
        gen = GENS[category]

        category_rows = []
        attempts = 0

        while len(category_rows) < target_count:
            attempts += 1
            formula = clean_expr(gen())

            if formula in seen:
                if attempts > target_count * 100:
                    break
                continue

            seen.add(formula)
            category_rows.append({
                "formula": formula,
                "source": "synthetic_simple",
                "category": category,
            })

        print(f"{category:15s}: {len(category_rows)}")
        rows.extend(category_rows)

    target_total = len(rows)
    print(f"final rows before shuffle: {target_total}")

    random.shuffle(rows)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["formula", "source", "category"],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {OUT}")
    print(f"unique formulas: {len(rows)}")
    print("note: actual count may be below N if some categories have limited unique space")


if __name__ == "__main__":
    main()