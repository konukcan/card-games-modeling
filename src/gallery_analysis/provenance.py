"""Provenance tracking for pipeline output files.

Computes deterministic hashes of probe sets, input files, and grammar
configurations so that output JSONs carry enough metadata to detect
whether two runs used the same inputs.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand


def compute_probe_hash(probes: List[Hand]) -> str:
    """SHA256 hash of a serialized probe set."""
    serialized = json.dumps([
        [(c.suit.value, c.rank.value) for c in hand]
        for hand in probes
    ], sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _file_hash(path: str) -> Optional[str]:
    """SHA256 hash of file contents, or None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_provenance(
    probe_seed: int,
    n_probes: int,
    probes: List[Hand],
    inject_path: Optional[str] = None,
    exemplar_path: Optional[str] = None,
    grammar_hash: Optional[str] = None,
    n_equiv_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a provenance dict for embedding in output JSON.

    All hash fields are computed from the actual objects/files used,
    not from user-provided strings. This ensures two result files
    can be compared to check whether they used the same inputs.

    Args:
        probe_seed: Random seed used for probe generation.
        n_probes: Number of probe hands.
        probes: The actual probe hands (hashed for verification).
        inject_path: Path to injection JSON file (hashed if exists).
        exemplar_path: Path to frozen-exemplars.json (hashed if exists).
        grammar_hash: Pre-computed hash of grammar productions.
        n_equiv_classes: Number of equivalence classes in the pool.

    Returns:
        Dict with provenance metadata.
    """
    return {
        "probe_seed": probe_seed,
        "n_probes": n_probes,
        "probe_hash": compute_probe_hash(probes),
        "inject_path": inject_path,
        "inject_hash": _file_hash(inject_path) if inject_path else None,
        "exemplar_path": exemplar_path,
        "exemplar_hash": _file_hash(exemplar_path) if exemplar_path else None,
        "grammar_hash": grammar_hash,
        "n_equiv_classes": n_equiv_classes,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
