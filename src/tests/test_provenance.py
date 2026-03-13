"""Tests for provenance tracking utility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gallery_analysis.provenance import compute_provenance, compute_probe_hash


class TestComputeProbeHash:
    def test_deterministic(self):
        """Same probe set produces same hash."""
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        h1 = compute_probe_hash(probes)
        h2 = compute_probe_hash(probes)
        assert h1 == h2

    def test_different_seeds_different_hash(self):
        """Different probe seeds produce different hashes."""
        from gallery_analysis.exemplars import generate_probe_set
        probes_a = generate_probe_set(10, seed=42)
        probes_b = generate_probe_set(10, seed=99)
        assert compute_probe_hash(probes_a) != compute_probe_hash(probes_b)

    def test_returns_hex_string(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        h = compute_probe_hash(probes)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex


class TestComputeProvenance:
    def test_returns_dict_with_required_keys(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        prov = compute_provenance(
            probe_seed=42,
            n_probes=10,
            probes=probes,
            n_equiv_classes=100,
        )
        assert "probe_seed" in prov
        assert "n_probes" in prov
        assert "probe_hash" in prov
        assert "n_equiv_classes" in prov
        assert "timestamp" in prov

    def test_optional_file_hashes(self):
        """When file paths provided, their hashes are included."""
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        # Use this test file as a stand-in for any real file
        test_file = Path(__file__)
        prov = compute_provenance(
            probe_seed=42, n_probes=10, probes=probes,
            inject_path=str(test_file),
        )
        assert "inject_hash" in prov
        assert prov["inject_hash"] is not None

    def test_missing_file_gives_none(self):
        from gallery_analysis.exemplars import generate_probe_set
        probes = generate_probe_set(10, seed=42)
        prov = compute_provenance(
            probe_seed=42, n_probes=10, probes=probes,
            inject_path="/nonexistent/file.json",
        )
        assert prov["inject_hash"] is None
