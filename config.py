"""NEMIX-v2 configuration.  All optimizations are flags, not hardcoded."""
from dataclasses import dataclass


@dataclass
class NEMIXConfig:
    # ---- vocabulary -------------------------------------------------
    vocab_size: int = 32256
    pad_id: int = 0
    mask_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    tokenizer_path: str = "deepseek-ai/deepseek-coder-6.7b-base"

    # ---- Gated DeltaNet backbone (20M target) ------------------------
    d_model: int = 288
    n_heads: int = 4
    n_layers: int = 4
    expand_v: float = 2.0
    use_gate: bool = True

    # ---- recurrent depth --------------------------------------------
    n_prelude: int = 1
    n_recurrent: int = 1
    n_coda: int = 1
    max_loop_iters: int = 2
    use_act: bool = False            # OFF at 20M: halting head adds noise
    act_threshold: float = 0.99
    act_penalty_weight: float = 1e-3

    # ---- any-process decoder ----------------------------------------
    n_dec_layers: int = 2
    block_size: int = 256
    refine_steps: int = 8
    commit_per_step: float = 0.15
    revise_margin: float = 0.05
    insert_alpha: float = 1.7
    max_insertions_per_step: int = 8
    delete_floor: float = 0.01

    # ---- MDM / Rao-Blackwellized loss -------------------------------
    elbo_max_weight: float = 5.0     # clamp on 1/alpha_t so gradients stay finite
    mask_warmup_start: float = 0.15  # initial t_max (15% mask, like BERT)
    mask_warmup_steps: int = 1000    # steps to reach t_max = 1.0
    aux_weight: float = 0.0          # insertion/deletion BCE off during base training

    # ---- gist memory ------------------------------------------------
    gist_dim: int = 128
    gist_capacity: int = 1024
    hier_group: int = 4
    hier_max_levels: int = 4
    gist_write_threshold: float = 0.90
    gist_block_tokens: int = 256
    salience_threshold: float = 4.0
    max_salient_tokens: int = 8

    # ---- MEGa continual learning ------------------------------------
    mega_rank: int = 4
    mega_max_adapters: int = 256
    mega_gate_temperature: float = 0.1
    mega_min_similarity: float = 0.70

    # ---- deliberation -------------------------------------------------
    n_drafts: int = 4
    verify_budget: int = 8

    # ---- device / dtype / optimization flags -------------------------
    device: str = "auto"
    dtype: str = "auto"
    use_compile: bool = False
    use_gradient_checkpointing: bool = False
    chunk_size: int = 128
    fused_norm: bool = False

    # ---- training ----------------------------------------------------
    lr: float = 1e-4                 # MDLM stable recipe (was 3e-4)
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0      # single clip for all params
    batch_size: int = 16
    grad_accum: int = 2
    max_seq_len: int = 512
    seed: int = 0