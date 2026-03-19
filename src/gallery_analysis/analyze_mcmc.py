"""
MCMC analysis pipeline: Metropolis-Hastings search over all 60 gallery rules.

This is the MCMC analog of analyze.py (which uses enumeration).
Both produce hypothesis pools for Bayesian scoring, enabling
direct comparison of the two search strategies.

Usage:
    cd src
    python -m gallery_analysis.analyze_mcmc [--n-steps 100000] [--n-chains 8] [--quick]

Examples:
    # Quick test (~5 min, 3 rules, 1000 steps, 2 chains)
    python -m gallery_analysis.analyze_mcmc --quick

    # Default run (~1-2 hours, all 60 rules)
    python -m gallery_analysis.analyze_mcmc --n-steps 100000 --n-chains 8

    # Full Piantadosi-scale run
    python -m gallery_analysis.analyze_mcmc --n-steps 250000 --n-chains 16
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import Arrow, BOOL, HAND
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.mcmc_search import (
    MCMCConfig, run_parallel_chains,
)
from gallery_analysis.mcmc_hypothesis_collector import TrajectoryAnalyzer


# ------------------------------------------------------------------ #
#  Quick-mode rule selection: one per difficulty group
# ------------------------------------------------------------------ #
QUICK_RULE_IDS = ['all_red', 'all_even', 'strict_increasing']


def run_mcmc_analysis(args: argparse.Namespace) -> None:
    """Main analysis pipeline.

    Steps:
        1. Build the gallery grammar (same grammar that enumeration uses).
        2. Load frozen exemplar hands for each rule.
        3. Determine which rules to analyze (all 60 or 3 in quick mode).
        4. Generate a shared probe set for extension-size estimation.
        5. Configure MCMC and run parallel chains for each rule.
        6. Analyze trajectories and collect summary statistics.
        7. Save results to JSON.
    """

    # 1. Build grammar (same as enumeration uses)
    grammar = build_gallery_grammar()
    print(f"Grammar: {len(grammar)} productions")

    # 2. Load exemplars
    exemplars = load_exemplars()
    print(f"Exemplars loaded: {len(exemplars)} rules")

    # 3. Determine which rules to analyze
    if args.quick:
        # Quick mode: one representative rule per difficulty group.
        # Filter to rules that exist in both GALLERY_RULES and exemplars.
        available = set(GALLERY_RULES.keys()) & set(exemplars.keys())
        rule_ids = [r for r in QUICK_RULE_IDS if r in available]
        if not rule_ids:
            # Fallback: just pick the first 3 available rules
            rule_ids = sorted(available)[:3]
            print(f"  (quick-mode fallback rules: {rule_ids})")
    else:
        # Full mode: intersection of GALLERY_RULES and exemplars
        rule_ids = sorted(set(GALLERY_RULES.keys()) & set(exemplars.keys()))

    print(f"Rules to analyze: {len(rule_ids)}")

    # 4. Generate shared probe set for extension estimation
    ext_probes = generate_probe_set(n_probes=10_000, seed=args.seed)

    # 5. Configure MCMC
    config = MCMCConfig(
        n_steps=args.n_steps,
        max_depth=args.max_depth,
        noise_epsilon=args.noise_epsilon,
        max_nodes=args.max_nodes,
        top_k=args.top_k,
        seed=args.seed,
        verbose=args.verbose,
        init_max_depth=args.init_max_depth,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    print(f"Config: {config.n_steps} steps x {args.n_chains} chains, "
          f"depth={config.max_depth} (init={config.init_max_depth}), "
          f"eps={config.noise_epsilon}, β={config.beta_start}→{config.beta_end}, "
          f"verbose={args.verbose}")
    print(f"{'='*60}")

    # 6. Run MCMC for each rule
    results: Dict[str, Any] = {}
    request_type = Arrow(HAND, BOOL)
    total_start = time.time()

    for i, rule_id in enumerate(rule_ids):
        rule_start = time.time()
        group = exemplars[rule_id]['group']
        print(f"\n[{i+1}/{len(rule_ids)}] {rule_id} (group {group})...")

        hands = exemplars[rule_id]['hands_primary']

        # Per-rule seed variation so different rules explore different trajectories
        rule_seed_offset = hash(rule_id) % 100_000

        # Run parallel chains
        result = run_parallel_chains(
            grammar, config,
            request_type=request_type,
            exemplar_hands=hands,
            n_chains=args.n_chains,
            ext_probe_hands=ext_probes,
            seed_offset=rule_seed_offset,
        )

        # Trajectory analysis
        analyzer = TrajectoryAnalyzer(result)
        summary = analyzer.summary()
        freq_ranking = analyzer.frequency_ranking(top_k=20)
        fp_ordering = analyzer.first_passage_ordering()

        rule_time = time.time() - rule_start

        results[rule_id] = {
            'rule_id': rule_id,
            'group': group,
            'answer': exemplars[rule_id]['answer'],
            'mcmc': {
                'n_steps': result.n_steps,
                'n_chains': args.n_chains,
                'n_accepted': result.n_accepted,
                'n_unique': result.n_unique,
                'acceptance_rate': result.acceptance_rate,
                'time_seconds': round(rule_time, 2),
            },
            'trajectory_summary': summary,
            'top_hypotheses': freq_ranking[:20],
            'first_passage_top10': fp_ordering[:10],
        }

        # Progress output
        print(f"  {result.n_unique} unique, accept={result.acceptance_rate:.3f}, "
              f"entropy={summary['entropy_bits']:.1f} bits, {rule_time:.1f}s")
        if freq_ranking:
            prog_str = freq_ranking[0]['program']
            display = prog_str[:80] + ('...' if len(prog_str) > 80 else '')
            print(f"  top: {display}")

        # At verbose >= 1, show top-5 hypotheses with details
        if args.verbose >= 1 and freq_ranking:
            print(f"\n  Top-5 hypotheses (by visit count):")
            for rank, hyp in enumerate(freq_ranking[:5], 1):
                visits = hyp['visit_count']
                emp_post = hyp['empirical_posterior']
                first = hyp['first_step']
                prog = hyp['program'][:100]
                print(f"    #{rank}: visits={visits} ({emp_post:.3f}) "
                      f"first@step={first}")
                print(f"        {prog}")

    total_time = time.time() - total_start

    # 7. Save results
    output = {
        'config': {
            'n_steps': args.n_steps,
            'n_chains': args.n_chains,
            'max_depth': args.max_depth,
            'noise_epsilon': args.noise_epsilon,
            'max_nodes': args.max_nodes,
            'top_k': args.top_k,
            'seed': args.seed,
        },
        'n_rules': len(rule_ids),
        'total_time_seconds': round(total_time, 2),
        'rules': results,
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = 'quick' if args.quick else f'{args.n_steps}steps_{args.n_chains}chains'
    output_path = results_dir / f'mcmc_{suffix}_{timestamp}_results.json'

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done! {len(rule_ids)} rules in {total_time:.0f}s")
    print(f"Results: {output_path}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run MCMC program search over gallery rules'
    )
    parser.add_argument('--n-steps', type=int, default=100_000,
                        help='MCMC steps per chain (default: 100000)')
    parser.add_argument('--n-chains', type=int, default=8,
                        help='Number of parallel chains (default: 8)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Maximum AST depth (default: 6)')
    parser.add_argument('--noise-epsilon', type=float, default=0.01,
                        help='Noise parameter for likelihood (default: 0.01)')
    parser.add_argument('--max-nodes', type=int, default=25,
                        help='Max program size in nodes (default: 25)')
    parser.add_argument('--top-k', type=int, default=250,
                        help='Top K hypotheses to retain (default: 250)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 3 rules, 1000 steps, 2 chains')
    parser.add_argument('--verbose', '-v', type=int, default=0,
                        help='Verbose level: 0=summary, 1=chain progress, '
                             '2=accept/reject, 3=proposal details')
    parser.add_argument('--beta-start', type=float, default=0.1,
                        help='Likelihood annealing start temperature (default: 0.1)')
    parser.add_argument('--beta-end', type=float, default=1.0,
                        help='Likelihood annealing end temperature (default: 1.0)')
    parser.add_argument('--init-max-depth', type=int, default=3,
                        help='Max depth for initial program sample (default: 3)')

    args = parser.parse_args()

    if args.quick:
        # Only override if user didn't explicitly provide values
        if '--n-steps' not in sys.argv:
            args.n_steps = 1000
        if '--n-chains' not in sys.argv:
            args.n_chains = 2

    run_mcmc_analysis(args)


if __name__ == '__main__':
    main()
