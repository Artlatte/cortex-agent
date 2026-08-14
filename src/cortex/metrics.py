"""Minimal dependency-free metrics registry with a Prometheus text exporter.

Counters, gauges and simple histogram summaries are all we need to answer the
questions production teams actually ask: how many LLM calls, how many retries,
what is the circuit state, how long do retrievals take.
"""

from __future__ import annotations

import threading
from collections import defaultdict


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: defaultdict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histogram_sums: defaultdict[str, float] = defaultdict(float)
        self._histogram_counts: defaultdict[str, int] = defaultdict(int)

    # -- counters ---------------------------------------------------------
    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def get_counter(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters[_key(name, labels)]

    # -- gauges -----------------------------------------------------------
    def set(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = value

    # -- histogram summary --------------------------------------------------
    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            k = _key(name, labels)
            self._histogram_sums[k] += value
            self._histogram_counts[k] += 1

    # -- rendering ----------------------------------------------------------
    def reset(self) -> None:
        """Clear every metric (mainly for tests)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histogram_sums.clear()
            self._histogram_counts.clear()

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return {
                **dict(self._counters),
                **dict(self._gauges),
                **{f"{k}_sum": v for k, v in self._histogram_sums.items()},
                **{f"{k}_count": v for k, v in self._histogram_counts.items()},
            }

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name.split('{')[0]} counter")
                lines.append(f"{name} {value:g}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name.split('{')[0]} gauge")
                lines.append(f"{name} {value:g}")
            for name, total in sorted(self._histogram_sums.items()):
                base = name.split("{")[0]
                lines.append(f"# TYPE {base} summary")
                lines.append(f"{name}_sum {total:g}")
                lines.append(f"{name}_count {self._histogram_counts[name]}")
        return "\n".join(lines) + ("\n" if lines else "")


METRICS = MetricsRegistry()
