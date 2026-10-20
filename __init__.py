"""NEMIX-v2: Gated DeltaNet backbone + any-process decoder + S*-style verification."""
from .config import NEMIXConfig
from .model import NEMIXModel
from .backbone import NEMIXBackbone, GDNBlock
from .refinement import AnyProcessDecoder, DecoderLayer
from .megas import MEGaBank, MEGaAdapter
from .sampler import (Sampler, compute_entropy, calibrate_temperature,
                      inverse_transform_sample, semantic_uniform_seeded)
from .memory import GistMemory
from .verifier import (arithmetic_verifier, confidence_verifier,
                       code_execution_verifier, pick_best)
from .creativity import (novelty_reweight, entropy_calibrated_sample,
                          recombine, divergence_ratio)
from .admission import (AdmissionSignals, ContinualMemory, admit,
                         admission_score, text_bloat,
                         observe_turn, consolidate_step)
from .deliberation import deliberate
from .dataset import build_dataset, load_training_pairs, split_train_val
from .train import (train_any, build_gpt_baseline, resolve_device, resolve_dtype,
                    evaluate_loss)

__all__ = [
    "NEMIXConfig", "NEMIXModel", "NEMIXBackbone", "GDNBlock",
    "AnyProcessDecoder", "DecoderLayer", "MEGaBank", "MEGaAdapter",
    "Sampler", "GistMemory", "AdmissionSignals", "ContinualMemory",
    "arithmetic_verifier", "confidence_verifier", "code_execution_verifier",
    "pick_best", "novelty_reweight", "entropy_calibrated_sample", "recombine",
    "divergence_ratio", "admit", "admission_score", "text_bloat",
    "observe_turn", "consolidate_step",
    "deliberate", "build_dataset", "load_training_pairs", "split_train_val",
    "train_any", "build_gpt_baseline", "resolve_device", "resolve_dtype",
    "evaluate_loss",
]