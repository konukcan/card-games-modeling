"""Tests for extension cache safety (probe hash validation)."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gallery_analysis.provenance import compute_probe_hash
from gallery_analysis.exemplars import generate_probe_set


class TestCacheSafety:
    def test_cache_with_valid_meta_is_used(self, tmp_path):
        """Cache with matching probe hash should be loaded normally."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        # Create a fake equivalence class
        equiv = [{
            "fingerprint": "fake_fp_001",
            "predicate": lambda h: True,
        }]

        # Write cache with valid _meta
        cache_file = tmp_path / "cache.json"
        cache_data = {
            "_meta": {
                "probe_seed": 42,
                "n_probes": 10,
                "probe_hash": probe_hash,
            },
            "fake_fp_001": [1000, 0.05],
        }
        cache_file.write_text(json.dumps(cache_data))

        # Should use cached value
        extensions = estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )
        assert extensions[0] == (1000, 0.05)

    def test_cache_with_wrong_meta_is_discarded(self, tmp_path):
        """Cache with wrong probe hash should be discarded entirely."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        equiv = [{
            "fingerprint": "fake_fp_001",
            "predicate": lambda h: len(h) > 0,
        }]

        # Write cache with WRONG probe hash
        cache_file = tmp_path / "cache.json"
        cache_data = {
            "_meta": {
                "probe_seed": 99,
                "n_probes": 10,
                "probe_hash": "wrong_hash_value",
            },
            "fake_fp_001": [9999, 0.99],
        }
        cache_file.write_text(json.dumps(cache_data))

        # Should discard cache and recompute
        extensions = estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )
        # Recomputed value should NOT be (9999, 0.99)
        assert extensions[0] != (9999, 0.99)

    def test_saved_cache_includes_meta(self, tmp_path):
        """After saving, cache file should include _meta block."""
        from gallery_analysis.analyze import estimate_extensions

        probes = generate_probe_set(10, seed=42)
        probe_hash = compute_probe_hash(probes)

        equiv = [{
            "fingerprint": "fp_test",
            "predicate": lambda h: True,
        }]

        cache_file = tmp_path / "cache.json"
        estimate_extensions(
            equiv, verbose=0, cache_path=str(cache_file),
            _probe_hash=probe_hash,
        )

        saved = json.loads(cache_file.read_text())
        assert "_meta" in saved
        assert saved["_meta"]["probe_hash"] == probe_hash
