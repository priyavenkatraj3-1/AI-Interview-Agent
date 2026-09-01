"""
Approximate USD cost-per-call estimation for Claude API usage, used to
populate CostLog rows (see backend/app/models/session.py).

Rates are Anthropic first-party per-million-token list prices (input,
output), matched by model-id prefix so a dated snapshot id (e.g.
"claude-haiku-4-5-20251001", as used by MODEL_CHEAP) resolves to the same
rate as the bare model id.
"""

_PRICING_TABLE_USD_PER_MTOK: list[tuple[str, float, float]] = [
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-sonnet-5", 2.00, 10.00),
]

# Conservative fallback if a model id doesn't match any known prefix.
_DEFAULT_RATE_USD_PER_MTOK = (3.00, 15.00)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    for prefix, input_rate, output_rate in _PRICING_TABLE_USD_PER_MTOK:
        if model.startswith(prefix):
            break
    else:
        input_rate, output_rate = _DEFAULT_RATE_USD_PER_MTOK
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
