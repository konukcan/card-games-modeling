"""Integration test: full pipeline with all robustness fixes."""
import sys
from pathlib import Path

# Ensure the src directory is on the import path so gallery_analysis is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.mark.slow
class TestPipelineIntegration:
    def test_quick_run_produces_valid_output(self):
        """A quick run should produce valid JSON with provenance.

        Uses the smallest viable parameters (depth=5, 50K programs) to keep
        wall-clock time reasonable (~2-3 minutes) while still exercising the
        full enumeration -> filter -> fingerprint -> score pipeline.
        """
        from gallery_analysis.analyze import run_analysis

        results = run_analysis(
            max_depth=5,
            max_programs=50_000,
            max_cost=25.0,
            timeout=120.0,
            extension_samples=10_000,
            verbose=0,
        )

        # -- Provenance block exists with probe_hash --
        assert "provenance" in results, "Results should contain a 'provenance' block"
        assert "probe_hash" in results["provenance"], (
            "Provenance should include 'probe_hash' for cache-safety validation"
        )

        # -- Rule results exist --
        assert len(results["rule_results"]) > 0, (
            "At least one gallery rule should be scored"
        )

        # -- Each rule result carries the approximate flag --
        for rule_id, rr in results["rule_results"].items():
            assert "true_rule_approximate" in rr, (
                f"Rule {rule_id} missing 'true_rule_approximate' key"
            )
