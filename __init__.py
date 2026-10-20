"""NEMIX-v3: Gated DeltaNet + any-process decoder + S*-style verification
+ edit-consistency loss + token-level halting + reliability-guided remasking
+ sparse episodic store + Engram hard-bypass + plasticity monitor
+ randomness-consistency loss + padding-free packing."""
from .config import NEMIXConfig
from .model import NEMIXModel
from .backbone import NEMIXBackbone, GDNBlock
from .refinement import AnyProcessDecoder, DecoderLayer
from .megas import MEGaBank, MEGaAdapter
from .sampler import (
    Sampler, compute_entropy, calibrate_temperature,
    inverse_transform_sample, semantic_uniform_seeded,
    detect_randomness_request, resolve_sampling_mode,
)
from .memory import GistMemory
from .verifier import (
    arithmetic_verifier, confidence_verifier,
    code_execution_verifier, pick_best,
)
from .creativity import (
    novelty_reweight, entropy_calibrated_sample,
    recombine, divergence_ratio,
)
from .admission import (
    AdmissionSignals, ContinualMemory, admit,
    admission_score, text_bloat,
    observe_turn, consolidate_step,
)
from .deliberation import deliberate
from .dataset import (
    build_dataset, load_training_pairs, split_train_val,
    pack_for_training,
)
from .train import (
    train_any, build_gpt_baseline, resolve_device, resolve_dtype,
    evaluate_loss,
)

# ---- new modules ----
from .plasticity import HistorySketch, ReadoutMonitor, adjust_theta_high
from .reliability import (
    TokenReliabilityHead, reliability_targets,
    leftward_relaxation_mask, select_remask_positions,
)
from .halting import (
    LoopMemoryAttention, HaltingPolicy,
    compute_oracle_halting_targets,
)
from .episodic import SparseEpisodicStore, EngramGate
from .randomness import (
    extract_valid_strings, strings_to_token_ids,
    randomness_consistency_loss, should_apply_randomness_loss,
)
from .packing import (
    pack_pairs, make_block_diag_attention_mask,
    VariableLengthStreamer,
)

__all__ = [
    # core
    "NEMIXConfig", "NEMIXModel", "NEMIXBackbone", "GDNBlock",
    "AnyProcessDecoder", "DecoderLayer", "MEGaBank", "MEGaAdapter",
    "Sampler", "GistMemory", "AdmissionSignals", "ContinualMemory",
    # verifiers
    "arithmetic_verifier", "confidence_verifier", "code_execution_verifier",
    "pick_best",
    # creativity
    "novelty_reweight", "entropy_calibrated_sample", "recombine",
    "divergence_ratio",
    # admission
    "admit", "admission_score", "text_bloat",
    "observe_turn", "consolidate_step",
    # deliberation
    "deliberate",
    # dataset
    "build_dataset", "load_training_pairs", "split_train_val",
    "pack_for_training",
    # training
    "train_any", "build_gpt_baseline", "resolve_device",
    "resolve_dtype", "evaluate_loss",
    # sampling helpers
    "detect_randomness_request", "resolve_sampling_mode",
    # new: plasticity
    "HistorySketch", "ReadoutMonitor", "adjust_theta_high",
    # new: reliability
    "TokenReliabilityHead", "reliability_targets",
    "leftward_relaxation_mask", "select_remask_positions",
    # new: halting
    "LoopMemoryAttention", "HaltingPolicy",
    "compute_oracle_halting_targets",
    # new: episodic
    "SparseEpisodicStore", "EngramGate",
    # new: randomness
    "extract_valid_strings", "strings_to_token_ids",
    "randomness_consistency_loss", "should_apply_randomness_loss",
    # new: packing
    "pack_pairs", "make_block_diag_attention_mask",
    "VariableLengthStreamer",
]