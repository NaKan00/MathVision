import csv
import random
from pathlib import Path


OUT = Path("datasets/school_symbols/formulas.csv")
N = 50000

VARS = ["x", "y", "z", "v", "r", "c", "a", "b", "m", "n", "t"]
DIGITS = list(range(0, 10))


def rv():
    return random.choice(VARS)


def ri(a=0, b=9):
    return random.randint(a, b)


def coef_var(power=None):
    k = ri(1, 9)
    x = rv()

    if power is None:
        power = random.choice([1, 2, 3])

    if power == 1:
        return f"{k}{x}"

    return f"{k}{x}^{{{power}}}"


def pythagorean():
    a, b, c = random.sample(VARS, 3)
    return f"{a}^{{2}}+{b}^{{2}}={c}^{{2}}"


def simple_power_eq():
    x, y = random.sample(VARS, 2)
    p = random.choice([2, 3])
    return f"{x}^{{{p}}}+{y}^{{{p}}}={ri(1, 9)}"


def v_over_c():
    return random.choice([
        r"\frac{v^{2}}{c^{2}}",
        r"1-\frac{v^{2}}{c^{2}}",
        r"\sqrt{1-\frac{v^{2}}{c^{2}}}",
        r"E=\frac{mc^{2}}{\sqrt{1-\frac{v^{2}}{c^{2}}}}",
        r"E_{0}=mc^{2}",
        r"E=mc^{2}",
    ])


def trig_digits():
    f = random.choice([r"\sin", r"\cos", r"\tan"])
    angle = random.choice([0, 15, 30, 45, 60, 90, 180])
    val = random.choice(["0", "1", r"\frac{1}{2}", r"\frac{\sqrt{3}}{2}"])
    return f"{f}({angle})={val}"


def trig_vars():
    x, y = random.sample(VARS, 2)
    return random.choice([
        rf"\sin{x}+\cos{y}=0",
        rf"\sin({x})+\cos({y})=0",
        rf"\cos{x}+\sin{y}=1",
        rf"\sin^{{2}}({x})+\cos^{{2}}({x})=1",
    ])


def fraction_symbols():
    a, b = random.sample(VARS, 2)
    return random.choice([
        rf"\frac{{{a}^{{2}}}}{{{b}^{{2}}}}",
        rf"\frac{{{a}+{b}}}{{{a}-{b}}}",
        rf"\frac{{{ri(1,9)}{a}}}{{{ri(1,9)}{b}}}",
        rf"{a}=\frac{{{b}^{{2}}}}{{{ri(1,9)}}}",
        rf"\frac{{{a}^{{2}}+{b}^{{2}}}}{{{ri(1,9)}}}",
    ])


def sqrt_symbols():
    a, b = random.sample(VARS, 2)
    return random.choice([
        rf"\sqrt{{{a}^{{2}}+{b}^{{2}}}}",
        rf"\sqrt{{{ri(1,9)}{a}^{{2}}+{ri(1,9)}{b}^{{2}}}}",
        rf"{a}=\sqrt{{{b}^{{2}}-{ri(1,9)}}}",
        rf"\sqrt{{1-\frac{{{a}^{{2}}}}{{{b}^{{2}}}}}}",
    ])


def indexed():
    a = rv()
    return random.choice([
        rf"{a}_{{1}}+{a}_{{2}}={a}_{{3}}",
        rf"{a}_{{n+1}}={a}_{{n}}+1",
        rf"{a}_{{0}}={ri(0,9)}",
        rf"{a}_{{i}}^{{2}}+{a}_{{j}}^{{2}}={a}_{{k}}^{{2}}",
    ])


def arithmetic():
    a, b, c = ri(1, 99), ri(1, 99), ri(1, 99)
    op = random.choice(["+", "-", r"\times", r"\div"])
    return f"{a}{op}{b}={c}"


def linear_eq():
    x = rv()
    a, b, c = ri(1, 9), ri(0, 9), ri(0, 99)
    sign = random.choice(["+", "-"])
    return f"{a}{x}{sign}{b}={c}"


def quadratic_eq():
    x = rv()
    a, b, c = ri(1, 9), ri(1, 9), ri(0, 9)
    sign1 = random.choice(["+", "-"])
    sign2 = random.choice(["+", "-"])
    return f"{a}{x}^{{2}}{sign1}{b}{x}{sign2}{c}=0"


def derivative_short():
    x = rv()
    p = random.choice([2, 3, 4])
    return rf"\frac{{d}}{{d{x}}}{x}^{{{p}}}={p}{x}^{{{p-1}}}"


def integral_short():
    x = rv()
    p = random.choice([1, 2, 3])
    return rf"\int {x}^{{{p}}}\,d{x}=\frac{{{x}^{{{p+1}}}}}{{{p+1}}}+C"


def generator():
    gens = [
        pythagorean,
        simple_power_eq,
        v_over_c,
        trig_digits,
        trig_vars,
        fraction_symbols,
        sqrt_symbols,
        indexed,
        arithmetic,
        linear_eq,
        quadratic_eq,
        derivative_short,
        integral_short,
    ]

    return random.choice(gens)()


def main():
    random.seed(123)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    formulas = set()

    while len(formulas) < N:
        formulas.add(generator())

        if len(formulas) % 5000 == 0:
            print("generated:", len(formulas))

    rows = list(formulas)
    random.shuffle(rows)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["formula"])
        writer.writeheader()

        for formula in rows:
            writer.writerow({"formula": formula})

    print("saved:", OUT)
    print("unique formulas:", len(rows))


if __name__ == "__main__":
    main()