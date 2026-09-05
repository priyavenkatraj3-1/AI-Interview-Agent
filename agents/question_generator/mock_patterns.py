"""
Per-pattern, procedurally-generated MCQ content for MockQuestionGeneratorAgent
(see question_generator.py).

Each function in PATTERN_GENERATORS takes (rng, difficulty, offset) and
returns (question_body, options, correct_option, explanation) for one
taxonomy pattern (agents.question_generator.taxonomy.APTITUDE_TAXONOMY).
Every value is computed from randomized inputs at call time -- there is no
bank of pre-written questions here, only the arithmetic/logical rules a real
placement-test question of that pattern follows. This is what MOCK_MODE
actually serves by default (no Anthropic credits required), so it needs to
cover the full taxonomy with real quantitative/logical/analytical content,
not a single addition problem reused for every topic.

`offset` is `len(previous_questions) % 9`: a small, bounded value baked into
one of each generator's numeric inputs so consecutive calls in the same
session (which only differ in how many questions have been asked so far)
are guaranteed to produce different numbers -- deterministically, not just
with high probability from the rng seed alone.
"""
from __future__ import annotations

import math
import random
from typing import Callable

Generated = tuple[str, list[str], int, str]
Generator = Callable[[random.Random, int, int], Generated]


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _int_mcq(rng: random.Random, correct: int, spread: int, min_value: int | None = None) -> tuple[list[str], int]:
    spread = max(2, spread)
    options = [correct]
    tries = 0
    while len(options) < 4 and tries < 200:
        tries += 1
        delta = rng.randint(-spread, spread)
        if delta == 0:
            continue
        candidate = correct + delta
        if min_value is not None and candidate < min_value:
            continue
        if candidate in options:
            continue
        options.append(candidate)
    while len(options) < 4:
        options.append(options[-1] + 1)
    rng.shuffle(options)
    return [str(o) for o in options], options.index(correct)


def _float_mcq(rng: random.Random, correct: float, spread: float, min_value: float = 0.0) -> tuple[list[str], int]:
    spread = max(0.5, spread)
    correct_r = round(correct, 2)
    values = [correct_r]
    tries = 0
    while len(values) < 4 and tries < 200:
        tries += 1
        delta = rng.uniform(-spread, spread)
        if abs(delta) < 0.05:
            continue
        candidate = round(correct_r + delta, 2)
        if candidate <= min_value or any(abs(candidate - v) < 0.01 for v in values):
            continue
        values.append(candidate)
    while len(values) < 4:
        values.append(round(values[-1] + 0.5, 2))
    rng.shuffle(values)
    return [_fmt_num(v) for v in values], values.index(correct_r)


# --------------------------------------------------------------------------
# Quantitative
# --------------------------------------------------------------------------


def gen_percentages(rng: random.Random, difficulty: int, offset: int) -> Generated:
    pct = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75])
    denom = 100 // math.gcd(100, pct)
    base = denom * (rng.randint(2, 5 + difficulty) + offset)
    result = base * pct // 100
    question = f"What is {pct}% of {base}?"
    options, idx = _int_mcq(rng, result, max(2, result // 8 or 2), min_value=0)
    explanation = f"{pct}% of {base} = ({pct}/100) x {base} = {result}."
    return question, options, idx, explanation


def gen_profit_and_loss(rng: random.Random, difficulty: int, offset: int) -> Generated:
    pct = rng.choice([5, 10, 12, 20, 25, 40, 50])
    denom = 100 // math.gcd(100, pct)
    cost = denom * (rng.randint(2, 6 + difficulty) + offset)
    is_profit = rng.choice([True, False])
    if is_profit:
        selling = cost + cost * pct // 100
        label = "profit"
    else:
        selling = cost - cost * pct // 100
        label = "loss"
    question = f"A shopkeeper buys an article for Rs {cost} and sells it for Rs {selling}. What is the {label} percentage?"
    options, idx = _int_mcq(rng, pct, max(2, pct // 4 or 2), min_value=0)
    explanation = f"{label.capitalize()} % = |{selling} - {cost}| / {cost} x 100 = {pct}%."
    return question, options, idx, explanation


def gen_interest(rng: random.Random, difficulty: int, offset: int) -> Generated:
    principal = 100 * (rng.randint(2, 8 + difficulty) + offset)
    if difficulty >= 4:
        rate, time = 10, 2
        ci = principal * 21 // 100
        question = (
            f"Find the compound interest on Rs {principal} at {rate}% per annum for {time} years, "
            "compounded annually."
        )
        options, idx = _int_mcq(rng, ci, max(2, ci // 6 or 2), min_value=0)
        explanation = (
            f"Amount = P(1+R/100)^T = {principal} x 1.1^2 = {principal + ci}. "
            f"CI = {principal + ci} - {principal} = {ci}."
        )
        return question, options, idx, explanation
    rate = rng.choice([4, 5, 6, 8, 10, 12])
    time = rng.randint(1, 3)
    si = principal * rate * time // 100
    question = f"Find the simple interest on Rs {principal} at {rate}% per annum for {time} year(s)."
    options, idx = _int_mcq(rng, si, max(2, si // 6 or 2), min_value=0)
    explanation = f"SI = (P x R x T)/100 = ({principal} x {rate} x {time})/100 = {si}."
    return question, options, idx, explanation


def gen_time_and_work(rng: random.Random, difficulty: int, offset: int) -> Generated:
    a = rng.randint(4, 10 + difficulty) + offset
    b = rng.randint(4, 12 + difficulty)
    days = round(a * b / (a + b), 2)
    question = (
        f"A can complete a piece of work in {a} days and B can complete it in {b} days. "
        "Working together, how many days will they take to complete the work?"
    )
    options, idx = _float_mcq(rng, days, max(0.5, days * 0.2))
    explanation = f"Combined time = (A x B)/(A + B) = ({a} x {b})/({a} + {b}) = {_fmt_num(days)} days."
    return question, options, idx, explanation


def gen_time_speed_and_distance(rng: random.Random, difficulty: int, offset: int) -> Generated:
    speed = rng.randint(30, 60 + difficulty * 5) + offset
    time_h = rng.choice([1, 2, 3, 4, 1.5, 2.5]) if difficulty >= 3 else rng.choice([1, 2, 3, 4])
    distance = round(speed * time_h, 2)
    ask = rng.choice(["distance", "time", "speed"])
    if ask == "distance":
        question = f"A train travels at a constant speed of {speed} km/h for {time_h} hours. What distance does it cover?"
        options, idx = _float_mcq(rng, distance, max(5, distance * 0.15))
        explanation = f"Distance = Speed x Time = {speed} x {time_h} = {_fmt_num(distance)} km."
    elif ask == "time":
        question = f"A car covers a distance of {distance} km at a speed of {speed} km/h. How long does the journey take, in hours?"
        options, idx = _float_mcq(rng, time_h, max(0.5, time_h * 0.2))
        explanation = f"Time = Distance / Speed = {distance} / {speed} = {_fmt_num(time_h)} hours."
    else:
        question = f"A cyclist covers {distance} km in {time_h} hours. What is the average speed, in km/h?"
        options, idx = _float_mcq(rng, speed, max(3, speed * 0.1))
        explanation = f"Speed = Distance / Time = {distance} / {time_h} = {_fmt_num(speed)} km/h."
    return question, options, idx, explanation


def gen_ratio_and_proportion(rng: random.Random, difficulty: int, offset: int) -> Generated:
    r1 = rng.randint(2, 6)
    r2 = rng.randint(2, 6)
    while r2 == r1:
        r2 = rng.randint(2, 6)
    multiplier = rng.randint(3, 8 + difficulty) + offset
    total = (r1 + r2) * multiplier
    share1, share2 = r1 * multiplier, r2 * multiplier
    first = rng.choice([True, False])
    if first:
        answer, ordinal = share1, "first"
        explanation = f"First share = {r1}/({r1}+{r2}) x {total} = {share1}."
    else:
        answer, ordinal = share2, "second"
        explanation = f"Second share = {r2}/({r1}+{r2}) x {total} = {share2}."
    question = f"Rs {total} is divided between two people in the ratio {r1}:{r2}. What is the {ordinal} person's share?"
    options, idx = _int_mcq(rng, answer, max(3, answer // 6 or 3), min_value=0)
    return question, options, idx, explanation


def gen_averages(rng: random.Random, difficulty: int, offset: int) -> Generated:
    n = rng.randint(4, 6 + difficulty // 2)
    base = rng.randint(10, 30) + offset
    nums = [base + rng.randint(0, 10) for _ in range(n)]
    total = sum(nums)
    avg = round(total / n, 2)
    question = f"Find the average of the following numbers: {', '.join(map(str, nums))}."
    options, idx = _float_mcq(rng, avg, max(1, avg * 0.15))
    explanation = f"Average = ({' + '.join(map(str, nums))}) / {n} = {_fmt_num(avg)}."
    return question, options, idx, explanation


def gen_number_system(rng: random.Random, difficulty: int, offset: int) -> Generated:
    if rng.choice(["remainder", "hcf"]) == "remainder":
        divisor = rng.choice([3, 4, 6, 7, 8, 9, 11, 12, 13])
        quotient = rng.randint(5, 15 + difficulty) + offset
        remainder = rng.randint(1, divisor - 1)
        n = divisor * quotient + remainder
        question = f"What is the remainder when {n} is divided by {divisor}?"
        options, idx = _int_mcq(rng, remainder, max(2, divisor // 2 or 2), min_value=0)
        explanation = f"{n} = {divisor} x {quotient} + {remainder}, so the remainder is {remainder}."
        return question, options, idx, explanation
    a_base = rng.randint(2, 6) + offset % 4
    b_base = rng.randint(2, 6)
    common = rng.randint(2, 5 + difficulty)
    a, b = common * a_base, common * b_base
    hcf = math.gcd(a, b)
    question = f"What is the HCF (Highest Common Factor) of {a} and {b}?"
    options, idx = _int_mcq(rng, hcf, max(2, hcf // 2 or 2), min_value=1)
    explanation = f"HCF of {a} and {b} is {hcf}."
    return question, options, idx, explanation


def gen_permutations_and_combinations(rng: random.Random, difficulty: int, offset: int) -> Generated:
    n = rng.randint(4, 6 + difficulty // 2) + (offset % 2)
    r = rng.randint(2, min(n, 4))
    if rng.choice(["combination", "permutation"]) == "combination":
        value = math.comb(n, r)
        question = f"In how many ways can {r} items be chosen from {n} distinct items (order does not matter)?"
        explanation = f"Number of ways = C({n},{r}) = {value}."
    else:
        value = math.perm(n, r)
        question = f"In how many ways can {r} items be arranged from {n} distinct items (order matters)?"
        explanation = f"Number of ways = P({n},{r}) = {value}."
    options, idx = _int_mcq(rng, value, max(2, value // 5 or 2), min_value=0)
    return question, options, idx, explanation


def gen_probability(rng: random.Random, difficulty: int, offset: int) -> Generated:
    red = rng.randint(2, 6) + offset % 3
    blue = rng.randint(2, 6)
    green = rng.randint(0, 4)
    total = red + blue + green
    choices = [("red", red), ("blue", blue)] + ([("green", green)] if green else [])
    color, count = rng.choice(choices)
    g = math.gcd(count, total)
    num, den = count // g, total // g
    answer = f"{num}/{den}"
    ball_desc = f"{red} red, {blue} blue" + (f" and {green} green" if green else "")
    question = f"A bag contains {ball_desc} balls. If one ball is drawn at random, what is the probability that it is {color}?"
    distractors: set[str] = set()
    while len(distractors) < 3:
        dn = rng.randint(1, den)
        dd = rng.randint(dn + 1, den + 3)
        g2 = math.gcd(dn, dd)
        candidate = f"{dn // g2}/{dd // g2}"
        if candidate != answer:
            distractors.add(candidate)
    options = [answer] + list(distractors)
    rng.shuffle(options)
    idx = options.index(answer)
    reduction = "" if f"{count}/{total}" == answer else f" = {answer}"
    explanation = f"P({color}) = {count}/{total}{reduction}."
    return question, options, idx, explanation


def gen_ages(rng: random.Random, difficulty: int, offset: int) -> Generated:
    b_age = rng.randint(8, 20) + offset
    diff = rng.randint(4, 15 + difficulty)
    a_age = b_age + diff
    years = rng.randint(2, 10)
    if rng.choice([True, False]):
        answer = a_age + years
        question = f"A is {diff} years older than B. B is currently {b_age} years old. What will A's age be after {years} years?"
        explanation = f"A's current age = {b_age} + {diff} = {a_age}. After {years} years, A's age = {a_age} + {years} = {answer}."
    else:
        answer = a_age
        question = f"A is {diff} years older than B. B is currently {b_age} years old. What is A's current age?"
        explanation = f"A's current age = B's age + {diff} = {b_age} + {diff} = {answer}."
    options, idx = _int_mcq(rng, answer, max(2, answer // 10 or 2), min_value=0)
    return question, options, idx, explanation


def gen_mixtures_and_alligation(rng: random.Random, difficulty: int, offset: int) -> Generated:
    conc_a = rng.choice([10, 20, 25, 30, 40])
    conc_b = rng.choice([v for v in [50, 60, 70, 80, 90] if v > conc_a])
    ratio_a = rng.randint(1, 3)
    ratio_b = rng.randint(1, 3)
    mixture_conc = round((conc_a * ratio_a + conc_b * ratio_b) / (ratio_a + ratio_b), 2)
    question = (
        f"Solution A has {conc_a}% acid concentration and Solution B has {conc_b}% acid concentration. "
        f"They are mixed in the ratio {ratio_a}:{ratio_b}. What is the acid concentration of the resulting mixture?"
    )
    options, idx = _float_mcq(rng, mixture_conc, max(2, mixture_conc * 0.1))
    explanation = (
        f"Mixture % = ({conc_a} x {ratio_a} + {conc_b} x {ratio_b}) / ({ratio_a} + {ratio_b}) = "
        f"{_fmt_num(mixture_conc)}%."
    )
    return question, options, idx, explanation


# --------------------------------------------------------------------------
# Logical reasoning (incl. analytical-reasoning-style patterns)
# --------------------------------------------------------------------------


def gen_number_series(rng: random.Random, difficulty: int, offset: int) -> Generated:
    rule = rng.choice(["arithmetic", "geometric", "squares", "increasing_diff"])
    length = 5
    if rule == "arithmetic":
        start = rng.randint(2, 9) + offset
        step = rng.randint(2, 5 + difficulty)
        seq = [start + i * step for i in range(length)]
        nxt = start + length * step
        rule_desc = f"add {step} each time"
    elif rule == "geometric":
        start = rng.randint(2, 4) + (offset % 3)
        ratio = rng.choice([2, 3])
        seq = [start * (ratio**i) for i in range(length)]
        nxt = start * (ratio**length)
        rule_desc = f"multiply by {ratio} each time"
    elif rule == "squares":
        start = rng.randint(2, 5) + (offset % 3)
        seq = [(start + i) ** 2 for i in range(length)]
        nxt = (start + length) ** 2
        rule_desc = "consecutive integers squared"
    else:
        start = rng.randint(2, 9) + offset
        step = rng.randint(2, 4 + difficulty)
        seq = [start]
        for _ in range(length - 1):
            seq.append(seq[-1] + step)
            step += 1
        nxt = seq[-1] + step
        rule_desc = "the difference between consecutive terms increases by 1 each time"
    question = f"Find the next number in the series: {', '.join(map(str, seq))}, ?"
    options, idx = _int_mcq(rng, nxt, max(2, nxt // 10 or 2))
    explanation = f"The pattern is: {rule_desc}. Next term = {nxt}."
    return question, options, idx, explanation


def gen_letter_series(rng: random.Random, difficulty: int, offset: int) -> Generated:
    step = rng.randint(1, 3 + difficulty // 2)
    start = (rng.randint(0, 25) + offset) % 26
    length = 5
    seq = [chr(65 + (start + i * step) % 26) for i in range(length)]
    nxt = chr(65 + (start + length * step) % 26)
    question = f"Find the next letter in the series: {', '.join(seq)}, ?"
    pool = [chr(65 + ((start + length * step + d) % 26)) for d in (-2, -1, 1, 2, 3)]
    options = [nxt]
    for p in pool:
        if p not in options:
            options.append(p)
        if len(options) == 4:
            break
    rng.shuffle(options)
    idx = options.index(nxt)
    explanation = f"Each letter advances by {step} position(s) in the alphabet. Next letter = {nxt}."
    return question, options, idx, explanation


_CODE_WORDS = ["CAT", "DOG", "SUN", "BALL", "FISH", "BIRD", "MOON", "STAR", "BOOK", "LAMP", "TREE", "GOLD"]


def gen_coding_decoding(rng: random.Random, difficulty: int, offset: int) -> Generated:
    def encode(word: str, shift: int) -> str:
        return "".join(chr((ord(c) - 65 + shift) % 26 + 65) for c in word)

    word = rng.choice(_CODE_WORDS)
    shift = 1 + (rng.randint(0, 2 + difficulty // 2) + offset) % 12
    coded = encode(word, shift)
    target_word = rng.choice([w for w in _CODE_WORDS if w != word])
    target_coded = encode(target_word, shift)
    question = f"In a certain code language, '{word}' is written as '{coded}'. How is '{target_word}' written in that code?"
    distractors: set[str] = set()
    while len(distractors) < 3:
        fake_shift = rng.choice([s for s in range(1, 15) if s != shift])
        candidate = encode(target_word, fake_shift)
        if candidate != target_coded:
            distractors.add(candidate)
    options = [target_coded] + list(distractors)
    rng.shuffle(options)
    idx = options.index(target_coded)
    explanation = f"Each letter is shifted forward by {shift} position(s) in the alphabet, so '{target_word}' becomes '{target_coded}'."
    return question, options, idx, explanation


_RELATION_TRIPLES = [
    ("father", "mother", "grandfather"),
    ("mother", "father", "grandmother"),
    ("father", "father", "grandfather"),
    ("mother", "mother", "grandmother"),
    ("brother", "father", "uncle"),
    ("sister", "mother", "aunt"),
]
_RELATION_POOL = ["father", "mother", "grandfather", "grandmother", "uncle", "aunt", "brother", "sister", "cousin", "nephew"]
_NAME_POOL = ["Arjun", "Bina", "Chetan", "Divya", "Esha", "Farhan", "Gita", "Hari"]


def gen_blood_relations(rng: random.Random, difficulty: int, offset: int) -> Generated:
    r1, r2, answer = rng.choice(_RELATION_TRIPLES)
    a, b, c = rng.sample(_NAME_POOL, 3)
    question = f"{a} is the {r1} of {b}. {b} is the {r2} of {c}. How is {a} related to {c}?"
    distractors = rng.sample([p for p in _RELATION_POOL if p != answer], 3)
    options = [answer] + distractors
    rng.shuffle(options)
    idx = options.index(answer)
    explanation = f"{a} is {b}'s {r1}, and {b} is {c}'s {r2}, so {a} is {c}'s {answer}."
    return question, options, idx, explanation


def gen_direction_sense(rng: random.Random, difficulty: int, offset: int) -> Generated:
    steps = 2 + min(3, difficulty // 2)
    directions = ["North", "East", "South", "West"]
    dir_idx = rng.randint(0, 3)
    x = y = 0
    parts = []
    for i in range(steps):
        dist = rng.randint(2, 10) + offset % 5
        d = directions[dir_idx]
        parts.append(f"{'starts walking' if i == 0 else 'then walks'} {dist} km {d.lower()}")
        if d == "North":
            y += dist
        elif d == "South":
            y -= dist
        elif d == "East":
            x += dist
        else:
            x -= dist
        dir_idx = (dir_idx + rng.choice([1, -1])) % 4
    final_dist = round(math.hypot(x, y), 2)
    question = "A person " + ", ".join(parts) + ". How far is he from the starting point, in km?"
    options, idx = _float_mcq(rng, final_dist, max(1, final_dist * 0.2 or 1))
    explanation = (
        f"Net displacement is ({x}, {y}). Distance from start = sqrt({x}^2 + {y}^2) = "
        f"sqrt({x * x} + {y * y}) = {_fmt_num(final_dist)} km."
    )
    return question, options, idx, explanation


_SYLLOGISM_TERMS = ["Cats", "Dogs", "Animals", "Doctors", "Engineers", "Teachers", "Students", "Pens", "Flowers", "Roses", "Tables", "Chairs"]
_SYLLOGISM_OPTIONS = [
    "Yes, the conclusion follows",
    "No, the conclusion does not follow",
    "Cannot be determined",
    "Follows only in some cases",
]


def gen_syllogism(rng: random.Random, difficulty: int, offset: int) -> Generated:
    a, b, c = rng.sample(_SYLLOGISM_TERMS, 3)
    valid = rng.choice([True, False])
    form = rng.choice(["barbara", "celarent", "darii"])
    if form == "barbara":
        premises = f"All {a} are {b}. All {b} are {c}."
        valid_conclusion = f"All {a} are {c}."
        invalid_conclusion = f"All {c} are {a}."
    elif form == "celarent":
        premises = f"All {a} are {b}. No {b} is {c}."
        valid_conclusion = f"No {a} is {c}."
        invalid_conclusion = f"Some {a} are {c}."
    else:
        premises = f"Some {a} are {b}. All {b} are {c}."
        valid_conclusion = f"Some {a} are {c}."
        invalid_conclusion = f"All {a} are {c}."
    conclusion = valid_conclusion if valid else invalid_conclusion
    question = (
        f"Statements: {premises}\nConclusion: {conclusion}\n"
        "Does the conclusion logically follow from the statements?"
    )
    answer = _SYLLOGISM_OPTIONS[0] if valid else _SYLLOGISM_OPTIONS[1]
    idx = _SYLLOGISM_OPTIONS.index(answer)
    explanation = (
        f"Given '{premises}', the only conclusion that necessarily follows is '{valid_conclusion}'. "
        f"The given conclusion {'matches it' if valid else 'does not match it'}, so: {answer}."
    )
    return question, list(_SYLLOGISM_OPTIONS), idx, explanation


_ODD_ONE_OUT_CATEGORIES = {
    "multiples of 3": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    "prime numbers": [2, 3, 5, 7, 11, 13, 17, 19, 23],
    "perfect squares": [4, 9, 16, 25, 36, 49, 64, 81],
    "even numbers": [2, 4, 6, 8, 10, 12, 14, 16],
}


def gen_odd_one_out(rng: random.Random, difficulty: int, offset: int) -> Generated:
    label, pool = rng.choice(list(_ODD_ONE_OUT_CATEGORIES.items()))
    three = rng.sample(pool, 3)
    odd_candidates = [n for n in range(2, 100) if n not in pool]
    odd_one = rng.choice(odd_candidates)
    items = three + [odd_one]
    rng.shuffle(items)
    idx = items.index(odd_one)
    question = f"Find the odd one out: {', '.join(map(str, items))}"
    explanation = f"All the others are {label} except {odd_one}."
    return question, [str(i) for i in items], idx, explanation


_SEAT_NAME_POOL = ["Amit", "Bela", "Chirag", "Divya", "Esha", "Farah", "Gopal"]


def gen_seating_arrangement(rng: random.Random, difficulty: int, offset: int) -> Generated:
    n = 5 if difficulty < 4 else 6
    order = rng.sample(_SEAT_NAME_POOL, n)
    position = rng.randint(0, n - 1)
    person = order[position]
    clues = [f"{order[i]} sits immediately to the left of {order[i + 1]}" for i in range(n - 1)]
    question = (
        f"{n} friends ({', '.join(order)}) sit in a row facing north. "
        + ". ".join(clues)
        + f". Who sits at position {position + 1} from the left?"
    )
    distractors = rng.sample([p for p in order if p != person], 3)
    options = [person] + distractors
    rng.shuffle(options)
    idx = options.index(person)
    explanation = f"From the clues, the seating order from left to right is: {', '.join(order)}. Position {position + 1} is {person}."
    return question, options, idx, explanation


def gen_puzzles(rng: random.Random, difficulty: int, offset: int) -> Generated:
    y = rng.randint(5, 15) + offset
    x = y + rng.randint(2, 8 + difficulty)
    s, d = x + y, x - y
    question = f"The sum of two numbers is {s} and their difference is {d}. What is the larger number?"
    options, idx = _int_mcq(rng, x, max(2, x // 8 or 2), min_value=0)
    explanation = f"Larger number = (Sum + Difference) / 2 = ({s} + {d}) / 2 = {x}."
    return question, options, idx, explanation


_DATA_SUFFICIENCY_OPTIONS = [
    "Statement I alone is sufficient, but Statement II alone is not",
    "Statement II alone is sufficient, but Statement I alone is not",
    "Both statements together are sufficient, but neither alone is sufficient",
    "Either statement alone is sufficient",
]


def gen_data_sufficiency(rng: random.Random, difficulty: int, offset: int) -> Generated:
    target = rng.randint(10, 40) + offset
    mode = rng.choice(["first_alone", "second_alone", "both_needed", "either_alone"])
    if mode == "first_alone":
        stmt1, stmt2 = f"I. x = {target}.", f"II. y = {rng.randint(1, 10)} (y is an unrelated quantity)."
        answer = _DATA_SUFFICIENCY_OPTIONS[0]
    elif mode == "second_alone":
        stmt1, stmt2 = f"I. y = {rng.randint(1, 10)} (y is an unrelated quantity).", f"II. x = {target}."
        answer = _DATA_SUFFICIENCY_OPTIONS[1]
    elif mode == "both_needed":
        y = rng.randint(2, 10)
        stmt1, stmt2 = f"I. x + y = {target + y}.", f"II. x - y = {target - y}."
        answer = _DATA_SUFFICIENCY_OPTIONS[2]
    else:
        stmt1, stmt2 = f"I. x = {target}.", f"II. 2x = {2 * target}."
        answer = _DATA_SUFFICIENCY_OPTIONS[3]
    question = f"What is the value of x?\n{stmt1}\n{stmt2}"
    idx = _DATA_SUFFICIENCY_OPTIONS.index(answer)
    explanation = f"x = {target}. {answer}."
    return question, list(_DATA_SUFFICIENCY_OPTIONS), idx, explanation


_CONCLUSION_NAME_POOL = ["Ravi", "Sita", "Mohan", "Geeta", "Karan", "Meena"]


def gen_statement_and_conclusion(rng: random.Random, difficulty: int, offset: int) -> Generated:
    a, b = rng.sample(_CONCLUSION_NAME_POOL, 2)
    a_val = rng.randint(20, 80) + offset
    b_val = a_val + rng.choice([d for d in range(-20, 21) if d != 0])
    statement = f"Statement: {a} scored {a_val} marks and {b} scored {b_val} marks out of 100."
    answer = f"{a} scored more marks than {b}." if a_val > b_val else f"{b} scored more marks than {a}."
    options = [
        f"{a} scored more marks than {b}.",
        f"{b} scored more marks than {a}.",
        f"{a} and {b} scored equal marks.",
        "The data given is insufficient to compare.",
    ]
    idx = options.index(answer)
    question = f"{statement}\nWhich of the following conclusions definitely follows?"
    explanation = f"{a} scored {a_val} and {b} scored {b_val}, so: {answer}"
    return question, options, idx, explanation


def gen_pattern_based_reasoning(rng: random.Random, difficulty: int, offset: int) -> Generated:
    op = "sum" if difficulty < 4 else "product"
    rows = []
    for _ in range(3):
        rx = rng.randint(1, 9) + offset % 5
        ry = rng.randint(1, 9)
        rz = rx + ry if op == "sum" else rx * ry
        rows.append((rx, ry, rz))
    missing_row = rng.randint(0, 2)
    x, y, z = rows[missing_row]
    lines = [f"{rx} , {ry} , {'?' if i == missing_row else rz}" for i, (rx, ry, rz) in enumerate(rows)]
    question = "Study the pattern in the grid below (each row follows the same rule):\n" + "\n".join(lines) + "\nWhat number replaces '?'"
    options, idx = _int_mcq(rng, z, max(2, z // 8 or 2), min_value=0)
    rule = "the sum of the first two numbers" if op == "sum" else "the product of the first two numbers"
    explanation = f"In each row, the third number is {rule}. So the missing value = {z}."
    return question, options, idx, explanation


def gen_data_interpretation(rng: random.Random, difficulty: int, offset: int) -> Generated:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    chosen = rng.sample(months, 4)
    values = [rng.randint(20, 100) + offset for _ in chosen]
    table = "\n".join(f"{m}: {v} units" for m, v in zip(chosen, values))
    prefix = f"The table below shows units sold each month:\n{table}\n"
    qtype = rng.choice(["total", "average", "difference", "max"])
    if qtype == "total":
        answer = sum(values)
        question = prefix + "What is the total number of units sold across all months?"
        explanation = f"Total = {' + '.join(map(str, values))} = {answer}."
        options, idx = _int_mcq(rng, answer, max(3, answer // 8 or 3), min_value=0)
    elif qtype == "average":
        answer = round(sum(values) / len(values), 2)
        question = prefix + "What is the average number of units sold per month?"
        explanation = f"Average = ({' + '.join(map(str, values))}) / {len(values)} = {_fmt_num(answer)}."
        options, idx = _float_mcq(rng, answer, max(3, answer * 0.15))
    elif qtype == "difference":
        hi, lo = max(values), min(values)
        answer = hi - lo
        question = prefix + "What is the difference between the highest and lowest monthly sales?"
        explanation = f"Difference = {hi} - {lo} = {answer}."
        options, idx = _int_mcq(rng, answer, max(2, answer // 4 or 2), min_value=0)
    else:
        best_month = chosen[values.index(max(values))]
        question = prefix + "In which month were the most units sold?"
        options = [best_month] + [m for m in chosen if m != best_month]
        rng.shuffle(options)
        idx = options.index(best_month)
        explanation = f"{best_month} had the highest sales, at {max(values)} units."
    return question, options, idx, explanation


# --------------------------------------------------------------------------
# Verbal ability
# --------------------------------------------------------------------------

_SYNONYM_PAIRS = [
    ("happy", "joyful"), ("angry", "furious"), ("big", "large"), ("small", "tiny"),
    ("fast", "quick"), ("brave", "courageous"), ("smart", "intelligent"), ("sad", "unhappy"),
    ("begin", "start"), ("end", "finish"), ("rich", "wealthy"), ("difficult", "hard"),
]
_ANTONYM_PAIRS = [
    ("happy", "sad"), ("big", "small"), ("fast", "slow"), ("brave", "cowardly"),
    ("rich", "poor"), ("difficult", "easy"), ("begin", "end"), ("hot", "cold"),
    ("light", "dark"), ("strong", "weak"), ("ancient", "modern"), ("expand", "contract"),
]
_WORD_POOL = sorted({w for pair in _SYNONYM_PAIRS + _ANTONYM_PAIRS for w in pair})


def gen_synonyms_antonyms(rng: random.Random, difficulty: int, offset: int) -> Generated:
    mode = rng.choice(["synonym", "antonym"])
    pairs = _SYNONYM_PAIRS if mode == "synonym" else _ANTONYM_PAIRS
    word, answer = rng.choice(pairs)
    question = f"Choose the word that is the closest {mode} of '{word}'."
    distractors = rng.sample([w for w in _WORD_POOL if w not in (word, answer)], 3)
    options = [answer] + distractors
    rng.shuffle(options)
    idx = options.index(answer)
    explanation = f"'{answer}' is the closest {mode} of '{word}'."
    return question, options, idx, explanation


_SENTENCE_ERROR_TEMPLATES = [
    ("He go to school every day.", "He goes to school every day."),
    ("She have finished her homework.", "She has finished her homework."),
    ("They was playing football.", "They were playing football."),
    ("I is going to the market.", "I am going to the market."),
    ("The dogs barks loudly.", "The dogs bark loudly."),
    ("He don't like coffee.", "He doesn't like coffee."),
]


def gen_sentence_correction(rng: random.Random, difficulty: int, offset: int) -> Generated:
    wrong, correct = rng.choice(_SENTENCE_ERROR_TEMPLATES)
    question = f'Choose the grammatically correct version of the sentence: "{wrong}"'
    others = [c for _, c in _SENTENCE_ERROR_TEMPLATES if c != correct]
    distractors = rng.sample(others, 3)
    options = [correct] + distractors
    rng.shuffle(options)
    idx = options.index(correct)
    explanation = f'The correct form is: "{correct}".'
    return question, options, idx, explanation


def gen_reading_comprehension(rng: random.Random, difficulty: int, offset: int) -> Generated:
    name = rng.choice(["Ravi", "Sita", "Aman", "Neha", "Kabir"])
    cities = ["Mumbai", "Delhi", "Chennai", "Pune", "Bengaluru"]
    city = rng.choice(cities)
    age = rng.randint(20, 35) + offset % 5
    years = rng.randint(2, 10)
    passage = f"{name} is {age} years old and lives in {city}. {name} has been working as a software engineer for the past {years} years."
    qtype = rng.choice(["age", "city", "years"])
    if qtype == "age":
        question = f"{passage}\nHow old is {name}?"
        options, idx = _int_mcq(rng, age, 3, min_value=15)
        answer = str(age)
    elif qtype == "city":
        question = f"{passage}\nWhich city does {name} live in?"
        options = [city] + rng.sample([c for c in cities if c != city], 3)
        rng.shuffle(options)
        idx = options.index(city)
        answer = city
    else:
        question = f"{passage}\nHow many years has {name} been working as a software engineer?"
        options, idx = _int_mcq(rng, years, 2, min_value=1)
        answer = str(years)
    explanation = f"According to the passage, the answer is {answer}."
    return question, options, idx, explanation


_PARA_SETS = [
    [
        "First, gather all the ingredients needed for the recipe.",
        "Next, preheat the oven to the required temperature.",
        "Then, mix the ingredients together in a bowl.",
        "Finally, bake the mixture until it is golden brown.",
    ],
    [
        "The company was founded in a small garage.",
        "It slowly grew as more customers began to trust the brand.",
        "Within a decade, it became a household name.",
        "Today, it is one of the largest companies in its industry.",
    ],
    [
        "The students arrived early for the science fair.",
        "They set up their projects on the assigned tables.",
        "Judges then walked around asking questions.",
        "Finally, the winners were announced at the closing ceremony.",
    ],
    [
        "Rain clouds gathered over the city in the afternoon.",
        "Within an hour, a heavy downpour began.",
        "Streets started to flood near the low-lying areas.",
        "By evening, the rain had stopped and the sun came out.",
    ],
]


def gen_para_jumbles(rng: random.Random, difficulty: int, offset: int) -> Generated:
    correct_order = rng.choice(_PARA_SETS)
    labeled = list(zip("ABCD", correct_order))
    shuffled = labeled[:]
    rng.shuffle(shuffled)
    while [l for l, _ in shuffled] == [l for l, _ in labeled]:
        rng.shuffle(shuffled)
    display = "\n".join(f"{l}. {s}" for l, s in shuffled)
    correct_sequence = "".join(l for l, _ in labeled)
    question = f"Arrange the following sentences in the correct logical order:\n{display}\nWhich sequence is correct?"
    wrong_sequences: set[str] = set()
    while len(wrong_sequences) < 3:
        perm = "".join(rng.sample("ABCD", 4))
        if perm != correct_sequence:
            wrong_sequences.add(perm)
    options = [correct_sequence] + list(wrong_sequences)
    rng.shuffle(options)
    idx = options.index(correct_sequence)
    explanation = f"The correct logical sequence is {correct_sequence}."
    return question, options, idx, explanation


_FILL_BLANK_TEMPLATES = [
    ("She is very ___ about her new job.", "excited", ["confused", "afraid", "reluctant"]),
    ("He was too ___ to finish the marathon.", "exhausted", ["excited", "curious", "confident"]),
    ("The teacher asked the students to ___ the assignment by Friday.", "submit", ["ignore", "postpone", "cancel"]),
    ("Despite the rain, the match ___ on schedule.", "continued", ["stopped", "vanished", "paused"]),
    ("The manager was ___ with the team's performance.", "satisfied", ["confused", "annoyed", "indifferent"]),
]


def gen_fill_in_the_blanks(rng: random.Random, difficulty: int, offset: int) -> Generated:
    sentence, answer, wrongs = rng.choice(_FILL_BLANK_TEMPLATES)
    question = f"Fill in the blank: {sentence}"
    options = [answer] + list(wrongs)
    rng.shuffle(options)
    idx = options.index(answer)
    explanation = f"'{answer}' best completes the sentence."
    return question, options, idx, explanation


_ERROR_SPOT_TEMPLATES = [
    ("He don't know the answer.", "don't", "doesn't"),
    ("She go to college by bus.", "go", "goes"),
    ("The books is on the table.", "is", "are"),
    ("I has completed my project.", "has", "have"),
    ("They was late for the meeting.", "was", "were"),
]


def gen_error_spotting(rng: random.Random, difficulty: int, offset: int) -> Generated:
    sentence, wrong_word, correct_word = rng.choice(_ERROR_SPOT_TEMPLATES)
    question = f'Identify the correction needed in this sentence: "{sentence}"'
    answer = f"'{wrong_word}' should be '{correct_word}'"
    others = [f"'{w}' should be '{c}'" for _, w, c in _ERROR_SPOT_TEMPLATES if w != wrong_word]
    distractors = rng.sample(others, 3)
    options = [answer] + distractors
    rng.shuffle(options)
    idx = options.index(answer)
    explanation = f"{answer}: the correct sentence should use '{correct_word}' instead of '{wrong_word}'."
    return question, options, idx, explanation


def gen_fallback(rng: random.Random, difficulty: int, offset: int) -> Generated:
    """Used only if a pattern name isn't in PATTERN_GENERATORS (never happens
    for taxonomy-driven calls; kept as a defensive net, not a real code path)."""
    a = rng.randint(2, 9 + difficulty) + offset
    b = rng.randint(2, 9 + difficulty)
    product = a * b
    question = f"What is {a} multiplied by {b}?"
    options, idx = _int_mcq(rng, product, max(2, product // 8 or 2), min_value=0)
    explanation = f"{a} x {b} = {product}."
    return question, options, idx, explanation


PATTERN_GENERATORS: dict[str, Generator] = {
    # Quantitative
    "percentages": gen_percentages,
    "profit_and_loss": gen_profit_and_loss,
    "simple_and_compound_interest": gen_interest,
    "time_and_work": gen_time_and_work,
    "time_speed_and_distance": gen_time_speed_and_distance,
    "ratio_and_proportion": gen_ratio_and_proportion,
    "averages": gen_averages,
    "number_system": gen_number_system,
    "permutations_and_combinations": gen_permutations_and_combinations,
    "probability": gen_probability,
    "ages": gen_ages,
    "mixtures_and_alligation": gen_mixtures_and_alligation,
    # Logical / analytical reasoning
    "number_series": gen_number_series,
    "letter_series": gen_letter_series,
    "coding_decoding": gen_coding_decoding,
    "blood_relations": gen_blood_relations,
    "direction_sense": gen_direction_sense,
    "syllogism": gen_syllogism,
    "odd_one_out": gen_odd_one_out,
    "seating_arrangement": gen_seating_arrangement,
    "puzzles": gen_puzzles,
    "data_sufficiency": gen_data_sufficiency,
    "statement_and_conclusion": gen_statement_and_conclusion,
    "pattern_based_reasoning": gen_pattern_based_reasoning,
    "data_interpretation": gen_data_interpretation,
    # Verbal ability
    "synonyms_antonyms": gen_synonyms_antonyms,
    "sentence_correction": gen_sentence_correction,
    "reading_comprehension": gen_reading_comprehension,
    "para_jumbles": gen_para_jumbles,
    "fill_in_the_blanks": gen_fill_in_the_blanks,
    "error_spotting": gen_error_spotting,
}


def generate_mock_question(topic: str, pattern: str, difficulty: int, previous_questions: list[str]) -> dict:
    """Build one self-consistent MCQ for `pattern`, derived entirely from
    randomized numbers/entities -- never a fixed question bank. `offset`
    (bounded by len(previous_questions)) guarantees consecutive calls in the
    same session produce different numbers even at the same
    topic/pattern/difficulty."""
    idx = len(previous_questions)
    rng = random.Random(f"{topic}:{pattern}:{difficulty}:{idx}")
    offset = idx % 9
    generator = PATTERN_GENERATORS.get(pattern, gen_fallback)
    body, options, correct_option, explanation = generator(rng, difficulty, offset)
    label = f"[{topic.replace('_', ' ').title()} / {pattern.replace('_', ' ').title()}, Difficulty {difficulty}] "
    return {
        "question": label + body,
        "options": options,
        "correct_option": correct_option,
        "explanation": explanation,
    }
