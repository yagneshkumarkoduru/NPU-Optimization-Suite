from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from bandwidth_estimator import BandwidthEstimator
from cost_model import ScheduleCostModel
from fusion_logic import FusionLogic
from graph_builder import load_workload
from memory_hierarchy import MemoryHierarchy
from penalty_tuner import AdaptivePenaltyRefinement
from quantum_interface import QuantumInterface
from schedule_analysis import ScheduleAnalysis
from schedule_explainer import ScheduleExplainer
from scheduling_engine import ScheduleResult, SchedulingEngine


def load_config(path: Path) -> Dict:
    raw_text = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "config.yaml uses non-JSON YAML syntax but PyYAML is not installed. "
                "Either install 'pyyaml' or keep config.yaml JSON-compatible."
            ) from exc
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a mapping/object.")
        return loaded


def rank_score(evaluation: Dict) -> float:
    return evaluation["breakdown"]["total_cost"] - 120.0 * evaluation.get("feasibility", 0.0)


def select_best_result(
    trials: int,
    base_seed: int,
    builder: Callable[[int], ScheduleResult],
    evaluate_with_penalties: Callable[[Sequence[str]], Dict],
) -> Tuple[ScheduleResult, Dict]:
    best_result: ScheduleResult | None = None
    best_eval: Dict | None = None
    best_rank = float("inf")
    trial_log = []

    for idx in range(trials):
        seed = base_seed + idx * 37
        result = builder(seed)
        evaluation = evaluate_with_penalties(result.order)
        score = rank_score(evaluation)
        trial_log.append(
            {
                "trial": idx + 1,
                "seed": seed,
                "rank_score": score,
                "cost": evaluation["breakdown"]["total_cost"],
                "feasibility": evaluation["feasibility"],
            }
        )
        if score < best_rank:
            best_rank = score
            best_result = result
            best_eval = evaluation

    if best_result is None or best_eval is None:
        raise RuntimeError("No trial produced a valid schedule result.")

    best_result.metadata = dict(best_result.metadata)
    best_result.metadata["trials"] = trial_log
    best_result.metadata["selected_rank_score"] = best_rank
    return best_result, best_eval


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    graph = load_workload(root / config["input"]["workload"])

    memory_hierarchy = MemoryHierarchy(config["hardware"])
    bandwidth_estimator = BandwidthEstimator(config["hardware"])
    fusion_logic = FusionLogic(config["fusion"])
    cost_model = ScheduleCostModel(
        memory_hierarchy=memory_hierarchy,
        bandwidth_estimator=bandwidth_estimator,
        fusion_logic=fusion_logic,
        weights=config["cost_weights"],
    )

    analysis = ScheduleAnalysis()
    explainer = ScheduleExplainer()

    def evaluate(order: Sequence[str], penalties: Dict[str, float] | None = None) -> Dict:
        return cost_model.evaluate(graph, order, penalties=penalties)

    base_seed = int(config["experiment"]["seed"])
    search_trials = int(config["experiment"].get("search_trials", 2))
    quantum_trials = int(config["experiment"].get("quantum_trials", 2))

    initial_penalties = dict(config["apr"]["initial_penalties"])
    baseline_penalties = dict(initial_penalties)

    def greedy_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.greedy(penalties=baseline_penalties)

    greedy, greedy_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 11,
        builder=greedy_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
    )

    def lookahead_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.lookahead(
            penalties=baseline_penalties,
            lookahead_depth=int(config["search"]["lookahead_depth"]),
            evaluator=lambda order: evaluate(order, baseline_penalties)["breakdown"]["total_cost"],
        )

    lookahead, lookahead_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 67,
        builder=lookahead_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
    )

    def beam_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.beam_search(
            penalties=baseline_penalties,
            beam_width=int(config["search"]["beam_width"]),
            evaluator=lambda order: evaluate(order, baseline_penalties)["breakdown"]["total_cost"],
        )

    beam, beam_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 101,
        builder=beam_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
    )

    def anneal_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.simulated_annealing(
            penalties=baseline_penalties,
            evaluator=lambda order: evaluate(order, baseline_penalties)["breakdown"]["total_cost"],
            iterations=int(config["search"]["annealing_iterations"]),
            start_temp=float(config["search"]["annealing_start_temp"]),
            end_temp=float(config["search"]["annealing_end_temp"]),
        )

    anneal, anneal_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 149,
        builder=anneal_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
    )

    def quantum_builder(seed: int) -> ScheduleResult:
        q = QuantumInterface(graph=graph, random_seed=seed)
        q_result = q.qaoa_refine(
            seed_order=anneal.order,
            evaluator=evaluate,
            penalties=baseline_penalties,
            layers=int(config["quantum"]["layers"]),
            iterations=int(config["quantum"]["iterations"]),
            walkers=int(config["quantum"].get("walkers", 6)),
        )
        return ScheduleResult(
            strategy=q_result.strategy,
            order=q_result.order,
            score=q_result.score,
            metadata=q_result.metadata,
        )

    quantum_baseline, quantum_eval = select_best_result(
        trials=quantum_trials,
        base_seed=base_seed + 191,
        builder=quantum_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
    )

    apr_warm = AdaptivePenaltyRefinement(initial_penalties=initial_penalties)
    warm_scheduler = SchedulingEngine(graph=graph, random_seed=base_seed + 509)
    apr_warm_sa = warm_scheduler.simulated_annealing(
        penalties=apr_warm.get(),
        evaluator=lambda order: evaluate(order, apr_warm.get())["breakdown"]["total_cost"],
        iterations=int(config["apr"]["classical_warmup_iterations"]),
        start_temp=float(config["search"]["annealing_start_temp"]),
        end_temp=float(config["search"]["annealing_end_temp"]),
    )

    warm_candidates = [lookahead.order, beam.order, anneal.order, apr_warm_sa.order]
    warm_logs = []
    for order in warm_candidates:
        e = evaluate(order, apr_warm.get())
        apr_warm.update(e["violation_rate"], e["cost_impact"])
        warm_logs.append(
            {
                "cost": e["breakdown"]["total_cost"],
                "feasibility": e["feasibility"],
                "penalties_after_update": apr_warm.get(),
            }
        )
    ranked_warm = []
    for order in warm_candidates:
        probe = evaluate(order, apr_warm.get())
        ranked_warm.append((-probe["feasibility"], probe["breakdown"]["total_cost"], order))
    ranked_warm.sort(key=lambda item: (item[0], item[1]))
    apr_seed_order = ranked_warm[0][2]

    def quantum_apr_builder(seed: int) -> ScheduleResult:
        q = QuantumInterface(graph=graph, random_seed=seed)
        apr_instance = AdaptivePenaltyRefinement(initial_penalties=apr_warm.get())
        q_result = q.qaoa_with_apr(
            seed_order=apr_seed_order,
            evaluator=evaluate,
            apr=apr_instance,
            rounds=int(config["apr"]["rounds"]),
            layers=int(config["quantum"]["layers"]),
            iterations_per_round=int(config["apr"]["iterations_per_round"]),
            walkers=int(config["quantum"].get("walkers", 6)),
        )
        metadata = dict(q_result.metadata)
        metadata["warmup_trace"] = warm_logs
        return ScheduleResult(
            strategy=q_result.strategy,
            order=q_result.order,
            score=q_result.score,
            metadata=metadata,
        )

    quantum_apr: ScheduleResult | None = None
    quantum_apr_eval: Dict | None = None
    quantum_apr_rank = float("inf")
    quantum_apr_trials = []

    for idx in range(quantum_trials):
        seed = base_seed + 239 + idx * 37
        candidate = quantum_apr_builder(seed)
        candidate_penalties = candidate.metadata.get("final_penalties", apr_warm.get())
        candidate_eval = evaluate(candidate.order, candidate_penalties)
        candidate_rank = rank_score(candidate_eval)
        quantum_apr_trials.append(
            {
                "trial": idx + 1,
                "seed": seed,
                "rank_score": candidate_rank,
                "cost": candidate_eval["breakdown"]["total_cost"],
                "feasibility": candidate_eval["feasibility"],
            }
        )
        if candidate_rank < quantum_apr_rank:
            quantum_apr_rank = candidate_rank
            quantum_apr = candidate
            quantum_apr_eval = candidate_eval

    if quantum_apr is None or quantum_apr_eval is None:
        raise RuntimeError("No Quantum + APR trial produced a valid schedule result.")
    quantum_apr.metadata = dict(quantum_apr.metadata)
    quantum_apr.metadata["trials"] = quantum_apr_trials
    quantum_apr.metadata["selected_rank_score"] = quantum_apr_rank

    results = {
        "Greedy": {"order": greedy.order, "metadata": greedy.metadata, "evaluation": greedy_eval},
        "Lookahead": {"order": lookahead.order, "metadata": lookahead.metadata, "evaluation": lookahead_eval},
        "Beam Search": {"order": beam.order, "metadata": beam.metadata, "evaluation": beam_eval},
        "Simulated Annealing": {"order": anneal.order, "metadata": anneal.metadata, "evaluation": anneal_eval},
        "Quantum (QAOA)": {
            "order": quantum_baseline.order,
            "metadata": quantum_baseline.metadata,
            "evaluation": quantum_eval,
        },
        "Quantum + APR": {"order": quantum_apr.order, "metadata": quantum_apr.metadata, "evaluation": quantum_apr_eval},
    }

    summaries = {name: analysis.summarize(name, payload["evaluation"]) for name, payload in results.items()}
    metric_table = analysis.comparison_table(summaries)

    explanation_blocks = []
    for name, payload in results.items():
        explanation_blocks.append(explainer.explain(name, payload, summaries[name]))
        explanation_blocks.append("")
    explanations_text = "\n".join(explanation_blocks).strip()

    output_dir = root / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "schedules.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_snapshot": config,
                "results": results,
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(metric_table)
        f.write("\n")

    with (output_dir / "explanations.txt").open("w", encoding="utf-8") as f:
        f.write(explanations_text)
        f.write("\n")

    print("Experiment completed.")
    print(metric_table)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
