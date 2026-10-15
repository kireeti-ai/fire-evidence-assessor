"""Leakage-controlled splitting: cluster near-duplicates + known groups, split by cluster."""
from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

import numpy as np

PARTITIONS = ("train", "val", "cal", "test")


def cluster_ids(hashes: Sequence[int], groups: Optional[Sequence[str]] = None,
                embeddings=None, max_hamming: int = 6, min_cosine: float = 0.97) -> np.ndarray:
    """Union-find over (a) pHash within `max_hamming`, (b) CLIP-embedding cosine >= `min_cosine`,
    (c) shared group id (video, camera, sequence, incident, session, location). O(n^2): fine for
    dataset curation scale; shard by source for very large sets."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    h = np.array([int(x) for x in hashes], dtype=np.uint64)
    chunk = 512
    for a in range(0, n, chunk):  # vectorised popcount over chunks; O(n^2) work, O(chunk*n) memory
        d = np.bitwise_count(h[a:a + chunk, None] ^ h[None, :])
        for i, j in zip(*np.where(d <= max_hamming)):
            if a + int(i) < int(j):
                union(a + int(i), int(j))
    if embeddings is not None:
        e = np.asarray(embeddings, float)
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        sim = e @ e.T
        for i, j in zip(*np.where(np.triu(sim >= min_cosine, 1))):
            union(int(i), int(j))
    if groups is not None:
        first = {}
        for i, g in enumerate(groups):
            if g is None or g == "":
                continue
            if g in first:
                union(i, first[g])
            else:
                first[g] = i
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


def assign_partitions(clusters: np.ndarray, fractions=(0.7, 0.1, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """Assign whole clusters to train/val/cal/test greedily towards the target fractions."""
    assert abs(sum(fractions) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    ids, counts = np.unique(clusters, return_counts=True)
    order = rng.permutation(len(ids))
    target = np.array(fractions) * len(clusters)
    filled = np.zeros(4)
    part_of = {}
    for k in order:
        p = int(np.argmax((target - filled) / target))
        part_of[ids[k]] = p
        filled[p] += counts[k]
    return np.array([PARTITIONS[part_of[c]] for c in clusters])


def verify_no_leakage(clusters: np.ndarray, partitions: np.ndarray) -> None:
    """Raise if any cluster appears in more than one partition."""
    for c in np.unique(clusters):
        if len(set(partitions[clusters == c])) > 1:
            raise AssertionError(f"cluster {c} spans partitions {set(partitions[clusters == c])}")


def manifest_hash(rows: Sequence[dict]) -> str:
    """Stable hash of a data manifest, recorded in every run config and artifact."""
    blob = json.dumps(sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
