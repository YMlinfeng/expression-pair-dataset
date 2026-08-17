"""Strict verifier for cross-identity same-expression video pairs.

The design premise is that expression equality must be *constructed* in a shared
latent space, never discovered by search: on the FEC benchmark, AU-distance
metrics reach 40.7-47.1% and emotion embeddings 53.3% at predicting which two of
three faces share an expression, against 87.5% for a median human rater. This
package therefore only ever *rejects* candidate pairs.
"""

__all__ = [
    "au",
    "audio",
    "augment",
    "calibrate",
    "descriptors",
    "identity",
    "landmarks",
    "neutral",
    "report",
    "scene",
    "verify",
]
