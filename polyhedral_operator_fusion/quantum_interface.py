from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from graph_builder import WorkloadGraph
from penalty_tuner import AdaptivePenaltyRefinement


@dataclass
class QuantumResult:
    strategy: str
    order: List[str]
    score: float
    metadata: Dict


class QuantumInterface:
    def __init__(self, graph: WorkloadGraph, random_seed: int = 101):
        self.graph = graph
        self.rng = random.Random(random_seed)

    def _valid(self, order: Sequence[str]) -> bool:
        return self.graph.is_valid_order(order)

    def _sample_neighbor(self, order: Sequence[str], exploration: float) -> List[str]:
        base = list(order)
        n = len(base)
        tries = max(14, int(65 * exploration))
        for _ in range(tries):
            move = self.rng.choice(["swap", "insert", "block"])
            proposal = list(base)

            if move == "swap":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                proposal[i], proposal[j] = proposal[j], proposal[i]
            elif move == "insert":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                value = proposal.pop(i)
                proposal.insert(j, value)
            else:
                if n < 4:
                    continue
                left = self.rng.randrange(0, n - 2)
                right = self.rng.randrange(left + 1, min(n, left + 5))
                proposal[left:right] = reversed(proposal[left:right])

            if self._valid(proposal):
                return proposal
        return base

    def _pairwise_pressure(self, order: Sequence[str]) -> float:
        pressure = 0.0
        for idx in range(len(order) - 1):
            left = self.graph.nodes[order[idx]]
            right = self.graph.nodes[order[idx + 1]]
            traffic_jump = abs((left.input_size + left.output_size) - (right.input_size + right.output_size))
            op_penalty = 0.4 if left.op_type == right.op_type else 0.0
            pressure += 0.02 * traffic_jump + op_penalty
        return pressure

    def _energy(
        self,
        order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        penalties: Dict[str, float] | None,
    ) -> Tuple[float, Dict]:
        if not self._valid(order):
            return float("inf"), {
                "breakdown": {"total_cost": float("inf")},
                "feasibility": 0.0,
                "latency_cycles": float("inf"),
            }
        eval_report = evaluator(order, penalties)
        base = eval_report["breakdown"]["total_cost"]
        pairwise = self._pairwise_pressure(order)
        feasibility_bonus = 40.0 * eval_report.get("feasibility", 0.0)
        energy = base + pairwise - feasibility_bonus
        return energy, eval_report

    def qaoa_refine(
        self,
        seed_order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        penalties: Dict[str, float] | None = None,
        layers: int = 2,
        iterations: int = 80,
        walkers: int = 6,
    ) -> QuantumResult:
        layers = max(1, layers)
        walkers = max(2, walkers)

        gamma = [self.rng.uniform(0.3, 1.2) for _ in range(layers)]
        beta = [self.rng.uniform(0.2, 1.0) for _ in range(layers)]

        chains = [list(seed_order)]
        for _ in range(walkers - 1):
            chains.append(self._sample_neighbor(seed_order, exploration=1.1))

        chain_energy = []
        chain_eval = []
        for order in chains:
            e, rep = self._energy(order, evaluator, penalties)
            chain_energy.append(e)
            chain_eval.append(rep)

        best_idx = min(range(len(chains)), key=lambda i: chain_energy[i])
        best_order = list(chains[best_idx])
        best_energy = chain_energy[best_idx]
        best_eval = chain_eval[best_idx]
        accepted = 0
        moves = 0

        for step in range(iterations):
            layer = step % layers
            local_accept = 0

            for w in range(walkers):
                moves += 1
                base_order = chains[w]
                exploration = max(0.2, min(2.4, 1.2 + beta[layer] - 0.55 * gamma[layer]))
                proposal = self._sample_neighbor(base_order, exploration)
                prop_energy, prop_eval = self._energy(proposal, evaluator, penalties)

                delta = prop_energy - chain_energy[w]
                temp = max(0.04, gamma[layer])
                accept_prob = 1.0 if delta <= 0 else math.exp(-delta / temp)
                if self.rng.random() < accept_prob:
                    chains[w] = proposal
                    chain_energy[w] = prop_energy
                    chain_eval[w] = prop_eval
                    accepted += 1
                    local_accept += 1

                if chain_energy[w] < best_energy:
                    best_energy = chain_energy[w]
                    best_order = list(chains[w])
                    best_eval = chain_eval[w]

            accept_ratio = local_accept / max(1, walkers)
            gamma[layer] = max(0.05, min(2.0, gamma[layer] * (0.97 + 0.06 * (1.0 - accept_ratio))))
            beta[layer] = max(0.05, min(2.0, beta[layer] * (0.95 + 0.08 * accept_ratio)))

        return QuantumResult(
            strategy="Quantum (QAOA)",
            order=best_order,
            score=best_eval["breakdown"]["total_cost"],
            metadata={
                "layers": layers,
                "iterations": iterations,
                "walkers": walkers,
                "acceptance_ratio": accepted / max(1, moves),
                "backend": "qaoa_style_multiwalker_simulator",
                "final_gamma": gamma,
                "final_beta": beta,
                "best_energy": best_energy,
                "final_breakdown": best_eval["breakdown"],
            },
        )

    def qaoa_with_apr(
        self,
        seed_order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        apr: AdaptivePenaltyRefinement,
        rounds: int = 4,
        layers: int = 2,
        iterations_per_round: int = 55,
        walkers: int = 6,
    ) -> QuantumResult:
        order = list(seed_order)
        best_order = list(seed_order)
        best_score = float("inf")
        round_trace = []

        for idx in range(rounds):
            penalties = apr.get()
            adaptive_iters = int(iterations_per_round * (1.0 + 0.15 * idx))
            round_result = self.qaoa_refine(
                seed_order=order,
                evaluator=evaluator,
                penalties=penalties,
                layers=layers,
                iterations=adaptive_iters,
                walkers=walkers,
            )
            round_eval = evaluator(round_result.order, penalties)
            score = round_eval["breakdown"]["total_cost"]
            order = list(round_result.order)

            apr_state = apr.update(round_eval["violation_rate"], round_eval["cost_impact"])
            round_trace.append(
                {
                    "round": idx + 1,
                    "score": score,
                    "feasibility": round_eval.get("feasibility", 0.0),
                    "penalties": apr_state.penalties,
                    "apr_signal": apr_state.signal_snapshot,
                    "violation_rate": round_eval["violation_rate"],
                }
            )
            if score < best_score:
                best_score = score
                best_order = list(round_result.order)

        return QuantumResult(
            strategy="Quantum + APR",
            order=best_order,
            score=best_score,
            metadata={
                "backend": "qaoa_style_multiwalker_simulator",
                "rounds": rounds,
                "layers": layers,
                "walkers": walkers,
                "round_trace": round_trace,
                "final_penalties": apr.get(),
            },
        )
