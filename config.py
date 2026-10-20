"""NEMIX-v3 configuration. All optimizations are flags, not hardcoded."""
from dataclasses import dataclass


@dataclass
class NEMIXConfig:
    # ---- vocabulary -------------------------------------------------
    vocab_size: int = 32264
    pad_id: int = 32257
    mask_id: int = 32256
    bos_id: int = 32258
    eos_id: int = 32259
    tokenizer_path: str = "deepseek-ai/deepseek-coder-6.7b-base"

    # ---- Gated DeltaNet backbone ------------------------------------
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    expand_v: float = 2.0
    use_gate: bool = True

    # ---- hybrid attention (GDN-2 style) -----------------------------
    # Insert one full-attention layer every N GDN blocks (0 = pure GDN).
    hybrid_attention_every: int = 0
    # Fixed-size LRU exact KV cache alongside the recurrent state.
    hippocampus_size: int = 0

    # ---- recurrent depth --------------------------------------------
    n_prelude: int = 1
    n_recurrent: int = 1
    n_coda: int = 1
    max_loop_iters: int = 2
    use_act: bool = False
    act_threshold: float = 0.99
    act_penalty_weight: float = 1e-3

    # ---- token-level halting (MoR / LoopFormer / RecurTrace) --------
    use_halting: bool = False
    halting_oracle_loops: int = 4
    halting_use_loop_memory: bool = True
    halting_compute_penalty: float = 1e-3

    # ---- any-process decoder ----------------------------------------
    n_dec_layers: int = 1
    block_size: int = 256
    refine_steps: int = 8
    commit_per_step: float = 0.15
    revise_margin: float = 0.05
    insert_alpha: float = 1.7
    max_insertions_per_step: int = 8
    decoder_dropout: float = 0.1
    delete_floor: float = 0.01

    # ---- token-reliability head (PRISM) -----------------------------
    use_reliability_head: bool = True
    reliability_head_weight: float = 0.1
    reliability_remask_top_frac: float = 0.25

    # ---- edit-consistency loss (PURE / T2M) -------------------------
    edit_weight_start: float = 0.0
    edit_weight_final: float = 0.5
    edit_weight_warmup_steps: int = 1000
    edit_leftward_relaxation: int = 3

    # ---- MDM / Rao-Blackwellized loss -------------------------------
    elbo_max_weight: float = 5.0
    elbo_min_weight: float = 1.5
    elbo_weight_warmup_steps: int = 1000
    mask_warmup_start: float = 0.15
    mask_warmup_steps: int = 500
    aux_weight: float = 0.0

    # ---- gist memory ------------------------------------------------
    gist_dim: int = 128
    gist_capacity: int = 1024
    hier_group: int = 4
    hier_max_levels: int = 4
    gist_write_threshold: float = 0.90
    gist_block_tokens: int = 256
    salience_threshold: float = 4.0
    max_salient_tokens: int = 8

    # ---- sparse episodic store (NoteBank) ---------------------------
    sparse_store_capacity: int = 4096
    sparse_store_path: str = ""
    sparse_min_novelty: float = 0.30
    sparse_replay_topk: int = 3

    # ---- Engram hard-bypass (AION) ----------------------------------
    use_engram_bypass: bool = True
    engram_confidence_threshold: float = 0.90
    engram_training_epochs: int = 12

    # ---- admission in training (OASIS + plasticity) -----------------
    admission_in_training: bool = True
    admission_history_size: int = 512
    readout_monitor_every: int = 500
    readout_rank_threshold: float = 0.05
    readout_partial_reset_frac: float = 0.02
    theta_high: float = 0.25
    theta_high_relaxed: float = 0.10
    theta_low: float = -0.05
    w_novelty: float = 0.25
    w_learnability: float = 0.25
    w_utility: float = 0.20
    w_redundancy: float = 0.15
    w_bloat: float = 0.30
    w_volatility: float = 0.15
    capacity_episodic: int = 64
    capacity_concepts: int = 128
    promote_hits: int = 2

    # ---- randomness-consistency loss -------------------------------
    randomness_consistency_weight: float = 0.05
    randomness_top_k: int = 10

    # ---- MEGa continual learning -----------------------------------
    mega_rank: int = 4
    mega_max_adapters: int = 256
    mega_gate_temperature: float = 0.1
    mega_min_similarity: float = 0.70

    # ---- deliberation -----------------------------------------------
    n_drafts: int = 4
    verify_budget: int = 8

    # ---- device / dtype / optimization flags -----------------------
    device: str = "auto"
    dtype: str = "auto"
    use_compile: bool = False
    use_gradient_checkpointing: bool = False
    chunk_size: int = 128
    fused_norm: bool = False

    # ---- training ---------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0
    batch_size: int = 16
    grad_accum: int = 1
    max_seq_len: int = 512
    seed: int = 0

    # ---- packing / variable length ---------------------------------
    use_packing: bool = True
    packing_max_len: int = 2048
    dataset_max_prompt_len: int = 0   # 0 = no truncation
    dataset_max_target_len: int = 0   # 0 = no truncation