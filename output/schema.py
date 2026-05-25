# output/schema.py
# Dataclass definitions for the Pair Health Object.

from dataclasses import dataclass, asdict


@dataclass
class PairHealthObject:
    pair: tuple
    date: str
    rai_a: float
    rai_b: float
    rai_divergence: float
    johansen_rank: int
    eigenvalue: float
    eigenvalue_trend: float   # 30d slope — negative = degrading
    half_life: float
    half_life_trend: float    # 30d slope — positive = slowing down
    zscore: float
    risk_flag: str            # NORMAL / WATCH / ELEVATED / SUSPEND
    signal_lag: int           # best Granger lag for this pair (pair-level, not time-varying)

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-friendly dict.
        Converts `pair` tuple to "A/B" string for clean serialisation.
        """
        d = asdict(self)
        # Convert pair tuple/list to a stable string representation
        if isinstance(d["pair"], (tuple, list)):
            d["pair"] = f"{d['pair'][0]}/{d['pair'][1]}"
        return d
