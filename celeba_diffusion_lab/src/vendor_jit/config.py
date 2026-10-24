"""Single entry point for configuration.

Every runtime knob of the repo lives in one of the dataclasses below, is parsed
from exactly one YAML entry point (:func:`load_config`) and is serialised back
into checkpoints verbatim.  No module is allowed to introduce magic constants
for shapes, STFT parameters or optimisation settings.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml

__all__ = [
    "AudioConfig",
    "STFTConfig",
    "MelFilterConfig",
    "ImageConfig",
    "ModelConfig",
    "FlowConfig",
    "DatasetSpec",
    "DataConfig",
    "StatsConfig",
    "OptimConfig",
    "TrainConfig",
    "SampleConfig",
    "EvalConfig",
    "TextConditionConfig",
    "Config",
    "ShapeSpec",
    "load_config",
    "config_from_dict",
    "canonical_hash",
]


# --------------------------------------------------------------------------------------
# leaf configs
# --------------------------------------------------------------------------------------
@dataclass
class AudioConfig:
    sample_rate: int = 24000
    segment_samples: int = 96000
    channels: int = 1

    def validate(self) -> None:
        if self.channels != 1:
            raise ValueError(f"only mono is supported, got channels={self.channels}")
        if self.sample_rate <= 0 or self.segment_samples <= 0:
            raise ValueError("sample_rate and segment_samples must be positive")


@dataclass
class STFTConfig:
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    window: str = "hann"
    center: bool = True
    pad_mode: str = "reflect"
    normalized: bool = False
    onesided: bool = True

    def validate(self) -> None:
        if not self.onesided:
            raise ValueError("only onesided=true STFT is supported by the Re/Im layout")
        if self.window != "hann":
            raise ValueError(f"unsupported window {self.window!r}; only 'hann' is implemented")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")

    @property
    def num_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def has_nyquist(self) -> bool:
        """True when the top frequency bin is structurally real (even n_fft)."""
        return self.n_fft % 2 == 0


@dataclass
class MelFilterConfig:
    """Slaney mel filterbank applied on top of the linear STFT magnitude.

    Defaults are frozen to match ``fast_token2wav/vq06_inpaint_wav/stft1024_audio.py``'s
    ``speecht5_logmel`` definition exactly (16 kHz target, 80 mel bins, fmin/fmax
    80/7600 Hz, slaney-normalized, non-HTK), so a checkpoint trained under
    ``representation: mel`` stays vocodable by the pretrained SpeechT5 HiFi-GAN.
    """

    n_mels: int = 80
    fmin: float = 80.0
    fmax: float = 7600.0
    log_floor: float = 1e-10
    htk: bool = False
    norm: str = "slaney"

    def validate(self) -> None:
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.fmin < 0.0 or self.fmax <= self.fmin:
            raise ValueError("require 0 <= fmin < fmax")
        if self.log_floor <= 0.0:
            raise ValueError("log_floor must be positive")
        if self.htk:
            raise ValueError("only the non-HTK slaney mel scale is implemented")
        if self.norm != "slaney":
            raise ValueError(f"unsupported mel filter norm {self.norm!r}; only 'slaney'")


@dataclass
class ImageConfig:
    """Plain RGB image shape for ``representation: "image"`` (e.g. CelebA-64).

    No STFT/mel machinery applies here: the network sees raw (normalized) pixels
    directly, so this only carries the shape the WebDataset loader decodes to.
    """

    height: int = 64
    width: int = 64
    channels: int = 3

    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("image.height/width must be positive")
        if self.channels not in (1, 3):
            raise ValueError(f"image.channels must be 1 or 3, got {self.channels}")


@dataclass
class ModelConfig:
    in_channels: int = 2
    patch_size: tuple[int, int] = (27, 16)  # [frequency, time]
    depth: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    bottleneck_dim: int = 128
    mlp_ratio: float = 4.0  # SwiGLU inner dim = round_to(2/3 * mlp_ratio * hidden, multiple_of)
    ffn_multiple_of: int = 8
    time_embed_scale: float = 1000.0
    use_sincos_pos: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    num_classes: int = 0
    class_dropout_prob: float = 0.1
    # Optional continuous vector conditioning (e.g. a masked multi-attribute
    # query), parallel to and independent of the integer num_classes/
    # class_embed pathway above -- see cond_sb's attribute-conditional
    # Schrodinger Bridge work, which reuses this model unchanged rather than
    # forking it. attr_cond_dim=0 (default) is a no-op: no parameter is even
    # created, so every existing config/checkpoint is untouched.
    attr_cond_dim: int = 0
    attr_dropout_prob: float = 0.0
    num_cond_tokens: int = 0
    dropout: float = 0.0
    # Optional post-unpatchify conv refinement head (disabled, 0, by default --
    # byte-identical architecture/state_dict to before for every existing config).
    # A large-patch patchified transformer captures global layout/color but not
    # local high-frequency texture (see docs/execplans/jit_celeba.md, following
    # arXiv:2511.18822's DiP); this lets the pixel-space grid mix across patch
    # boundaries after unpatchify instead of predicting every patch independently.
    output_conv_layers: int = 0
    output_conv_hidden: int = 64
    output_conv_kernel: int = 3

    def validate(self) -> None:
        if self.in_channels not in (1, 2, 3):
            raise ValueError(
                "in_channels must be 2 (Re/Im complex STFT), 1 (a real-valued "
                "spectrogram such as log-mel) or 3 (RGB image); the exact match "
                "against Config.representation is cross-checked in Config.validate()"
            )
        if len(self.patch_size) != 2:
            raise ValueError("patch_size must be [freq, time]")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = self.hidden_dim // self.num_heads
        if self.use_rope and head_dim % 4 != 0:
            raise ValueError(
                "rectangular 2D RoPE splits head_dim into two axes of paired dims; "
                f"head_dim={head_dim} must be divisible by 4"
            )
        if self.num_cond_tokens < 0 or self.num_classes < 0:
            raise ValueError("num_cond_tokens/num_classes must be non-negative")
        if self.attr_cond_dim < 0:
            raise ValueError("attr_cond_dim must be non-negative")
        if not 0.0 <= self.attr_dropout_prob <= 1.0:
            raise ValueError("attr_dropout_prob must be in [0, 1]")
        if self.output_conv_layers < 0:
            raise ValueError("output_conv_layers must be non-negative")
        if self.output_conv_layers > 0:
            if self.output_conv_hidden <= 0:
                raise ValueError("output_conv_hidden must be positive")
            if self.output_conv_kernel <= 0 or self.output_conv_kernel % 2 == 0:
                raise ValueError("output_conv_kernel must be a positive odd number")
        if not 0.0 <= self.class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be in [0, 1]")


@dataclass
class FlowConfig:
    p_mean: float = -0.8
    p_std: float = 0.8
    t_eps: float = 0.05
    noise_scale: float = 1.0

    def validate(self) -> None:
        if self.t_eps <= 0.0:
            raise ValueError("t_eps must be positive")
        if self.p_std <= 0.0:
            raise ValueError("p_std must be positive")


@dataclass
class DatasetSpec:
    key: str = ""
    input_root: str = ""
    manifest_jsonl: str = ""
    class_id: int = 0


_IMAGE_RESAMPLE_FILTERS = ("nearest", "bilinear", "bicubic", "lanczos")


@dataclass
class DataConfig:
    datasets: list[DatasetSpec] = field(default_factory=list)
    manifest: str = ""  # repo-native manifest produced by cli.build_manifest
    image_root: str = ""  # WebDataset shard tree root, only used by representation="image"
    # WebdatasetImageDataset decode/resize knobs (representation="image" only). These
    # are deliberately *not* part of Config.representation_signature(): they describe
    # how a raw file on disk is turned into the [C,H,W] tensor the codec sees, not the
    # normalized representation's shape/stats semantics, so changing them never
    # invalidates an existing stats artifact's hash gate for a dataset whose on-disk
    # geometry hasn't changed (e.g. adding an unused-by-default resize path here must
    # not perturb the CelebA-64 representation_hash -- see ImageConfig for the fields
    # that *do* participate in that hash).
    image_ext: str = "png"  # file extension (no dot) of each shard member, e.g. "jpg"
    # 0 => same as image.height/width, i.e. the loader requires an exact match and
    # never resizes (byte-identical to the pre-resize-support behaviour). Set both to
    # the on-disk decode size (e.g. 112) when it differs from image.height/width (e.g.
    # 96 for the 112->96 ArcFace rule) to enable a deterministic resize in __getitem__.
    image_source_height: int = 0
    image_source_width: int = 0
    # Only consulted when a resize is actually performed (source != target size).
    image_resize_filter: str = "bicubic"
    # Only consulted when image_root is an "oss://bucket/prefix" URI (see
    # data/arc2face_oss.py) -- CelebA's local WebDataset tar tree ignores this.
    # Credentials are never stored in config: resolved at runtime from
    # OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET or ~/.ossutilconfig.
    oss_endpoint: str = ""
    split_salt: str = "jit-stft-v1"
    val_fraction: float = 0.005
    test_fraction: float = 0.005
    min_duration_sec: float = 0.4
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    # runtime resampling backend; the repo ships exactly one implementation
    resample_backend: str = "polyphase_sinc"
    resample_lowpass_width: int = 32

    def validate(self) -> None:
        if self.resample_backend != "polyphase_sinc":
            raise ValueError(f"unknown resample backend {self.resample_backend!r}")
        if not 0.0 <= self.val_fraction < 1.0 or not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("val/test fractions must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")
        if not self.image_ext or self.image_ext.startswith("."):
            raise ValueError("image_ext must be a bare extension without a leading dot")
        if self.image_source_height < 0 or self.image_source_width < 0:
            raise ValueError("image_source_height/width must be non-negative")
        if (self.image_source_height == 0) != (self.image_source_width == 0):
            raise ValueError(
                "image_source_height and image_source_width must both be 0 (no "
                "resize) or both set (resize from that source size)"
            )
        if self.image_resize_filter not in _IMAGE_RESAMPLE_FILTERS:
            raise ValueError(
                f"image_resize_filter must be one of {_IMAGE_RESAMPLE_FILTERS}, "
                f"got {self.image_resize_filter!r}"
            )


@dataclass
class StatsConfig:
    path: str = ""
    max_segments: int = 20000
    batch_size: int = 32
    num_workers: int = 8
    sigma_floor_rel: float = 1e-3
    seed: int = 1234

    def validate(self) -> None:
        if self.sigma_floor_rel <= 0.0:
            raise ValueError("sigma_floor_rel must be positive")


@dataclass
class OptimConfig:
    name: str = "adam"
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    lr: float | None = None  # explicit absolute LR wins over linear scaling
    lr_at_reference_batch: float = 2.0e-4
    reference_global_batch: int = 1024
    schedule: str = "constant_after_warmup"
    warmup_steps: int = 2000
    grad_clip: float | None = None

    def validate(self) -> None:
        if self.name != "adam":
            raise ValueError(f"unsupported optimizer {self.name!r}")
        if self.schedule != "constant_after_warmup":
            raise ValueError(f"unsupported schedule {self.schedule!r}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

    def resolve_lr(self, global_batch: int) -> float:
        if self.lr is not None:
            return float(self.lr)
        return float(self.lr_at_reference_batch) * global_batch / float(self.reference_global_batch)


@dataclass
class TrainConfig:
    out_dir: str = "runs/debug"
    run_name: str = "debug"
    micro_batch: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 1000
    precision: str = "bf16"
    ema_decays: tuple[float, ...] = (0.9996, 0.9999)
    seed: int = 1234
    log_every: int = 20
    ckpt_every: int = 2000
    keep_last_ckpts: int = 3
    val_every: int = 2000
    val_batches: int = 20
    sample_every: int = 0
    num_t_buckets: int = 10
    compile: bool = False
    allow_tf32: bool = True
    dataloader_drop_last: bool = True

    def validate(self) -> None:
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"unsupported precision {self.precision!r}")
        if self.micro_batch <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("micro_batch and grad_accum_steps must be positive")
        if len(self.ema_decays) == 0:
            raise ValueError("at least one EMA decay is required")


@dataclass
class SampleConfig:
    num_samples: int = 4
    batch_size: int = 4
    num_steps: int = 50
    method: str = "heun"
    cfg_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.1, 1.0)
    seed: int = 0
    ema_index: int = 0
    class_ids: list[int] = field(default_factory=list)
    out_dir: str = ""

    def validate(self) -> None:
        if self.method not in {"euler", "heun"}:
            raise ValueError(f"unsupported sampler {self.method!r}")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")


@dataclass
class EvalConfig:
    num_real: int = 256
    num_fake: int = 256
    batch_size: int = 8
    nn_probe_size: int = 512
    out_dir: str = ""


@dataclass
class TextConditionConfig:
    """Cross-attention text conditioning on a frozen pretrained LLM's hidden states.

    Disabled (``enabled=False``) by default and additive everywhere it touches:
    a disabled config produces byte-identical model architecture and config
    hashes to a plain (non-text) run, so existing reim_stft/mel checkpoints are
    completely unaffected. See docs/execplans/jit_tts_genshin.md.
    """

    enabled: bool = False
    encoder_path: str = ""  # frozen text encoder, e.g. a local Qwen checkpoint dir
    encoder_hidden_size: int = 2560
    max_text_tokens: int = 48
    cache_dir: str = ""  # packed-shard text-embedding cache root (see cli.extract_text_embeddings)
    drop_prob: float = 0.1  # CFG dropout probability, mirrors model.class_dropout_prob
    seed: int = 1234

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.encoder_path:
            raise ValueError("text.encoder_path is required when text.enabled")
        if self.encoder_hidden_size <= 0:
            raise ValueError("text.encoder_hidden_size must be positive")
        if self.max_text_tokens <= 0:
            raise ValueError("text.max_text_tokens must be positive")
        if not self.cache_dir:
            raise ValueError("text.cache_dir is required when text.enabled")
        if not 0.0 <= self.drop_prob <= 1.0:
            raise ValueError("text.drop_prob must be in [0, 1]")


REPRESENTATIONS = ("reim_stft", "mel", "image")


@dataclass
class Config:
    # "reim_stft": the AGENTS.md baseline, network sees [B,2,F,T] Re/Im STFT.
    # "mel": a real-valued [B,1,n_mels,T] log-mel image, aligned with
    # fast_token2wav's speecht5_logmel + the pretrained SpeechT5 HiFi-GAN vocoder.
    # "image": a real-valued [B,3,H,W] RGB image (e.g. CelebA-64); see
    # docs/execplans/jit_celeba.md. No STFT/mel/audio config applies.
    representation: str = "reim_stft"
    audio: AudioConfig = field(default_factory=AudioConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    mel: MelFilterConfig = field(default_factory=MelFilterConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    text: TextConditionConfig = field(default_factory=TextConditionConfig)

    def validate(self) -> Config:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {self.representation!r}"
            )
        for name in (
            "audio",
            "stft",
            "mel",
            "image",
            "model",
            "flow",
            "data",
            "stats",
            "optim",
            "train",
            "sample",
            "text",
        ):
            getattr(self, name).validate()
        expected_channels = {"mel": 1, "image": self.image.channels}.get(self.representation, 2)
        if self.model.in_channels != expected_channels:
            raise ValueError(
                f"representation={self.representation!r} requires model.in_channels="
                f"{expected_channels}, got {self.model.in_channels}"
            )
        self.shapes  # noqa: B018 - triggers patch divisibility validation
        return self

    @property
    def shapes(self) -> ShapeSpec:
        return ShapeSpec.from_config(self)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def representation_signature(self) -> dict[str, Any]:
        """The subset of the config that normalization statistics depend on."""
        if self.representation == "image":
            # audio/stft are meaningless for raw pixels; keying on them would make
            # the stats hash churn on unrelated audio-field edits.
            return {
                "representation": "image",
                "image": dataclasses.asdict(self.image),
                "sigma_floor_rel": self.stats.sigma_floor_rel,
            }
        sig: dict[str, Any] = {
            "audio": dataclasses.asdict(self.audio),
            "stft": dataclasses.asdict(self.stft),
            "sigma_floor_rel": self.stats.sigma_floor_rel,
        }
        # Keyed only when non-default so the reim_stft hash is byte-identical to
        # before mel support existed; already-saved reim_stft stats/checkpoints on
        # the cluster stay valid without recomputation.
        if self.representation != "reim_stft":
            sig["representation"] = self.representation
            sig["mel"] = dataclasses.asdict(self.mel)
        return sig

    @property
    def representation_hash(self) -> str:
        return canonical_hash(self.representation_signature())

    def text_signature(self) -> dict[str, Any]:
        """The subset of the config a text-embedding cache artifact depends on."""
        return dataclasses.asdict(self.text)

    @property
    def text_hash(self) -> str:
        return canonical_hash(self.text_signature())


@dataclass(frozen=True)
class ShapeSpec:
    """All tensor shapes are derived here; nothing downstream hardcodes them."""

    num_freqs: int
    num_frames: int
    freq_pad: int
    time_pad: int
    patch_freq: int
    patch_time: int
    grid_freq: int
    grid_time: int
    channels: int = 2

    @property
    def num_patches(self) -> int:
        return self.grid_freq * self.grid_time

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_freq * self.patch_time

    @staticmethod
    def from_config(cfg: Config) -> ShapeSpec:
        if cfg.representation == "image":
            cfg.image.validate()
            num_freqs, num_frames = cfg.image.height, cfg.image.width
        else:
            cfg.audio.validate()
            cfg.stft.validate()
            # The STFT frame count only depends on n_fft/hop/center/segment length;
            # for representation="mel" the mel filterbank projects those frames down
            # to n_mels bins, so only the frequency axis size changes below.
            num_freqs = cfg.mel.n_mels if cfg.representation == "mel" else cfg.stft.num_freqs
            if cfg.stft.center:
                num_frames = 1 + cfg.audio.segment_samples // cfg.stft.hop_length
            else:
                # torch.stft's effective frame width with center=False is governed
                # by n_fft (the window is embedded in an n_fft-wide frame), not
                # win_length -- see docs/current_implementation_review_zh.md C-03.
                num_frames = (
                    1 + (cfg.audio.segment_samples - cfg.stft.n_fft) // cfg.stft.hop_length
                )
        if num_frames <= 0:
            raise ValueError("segment is shorter than one STFT frame")
        p_f, p_t = int(cfg.model.patch_size[0]), int(cfg.model.patch_size[1])
        if p_f <= 0 or p_t <= 0:
            raise ValueError("patch_size entries must be positive")
        freq_pad = int(math.ceil(num_freqs / p_f) * p_f)
        time_pad = int(math.ceil(num_frames / p_t) * p_t)
        if freq_pad % p_f != 0 or time_pad % p_t != 0:
            raise ValueError("padded grid is not divisible by the patch size")
        return ShapeSpec(
            num_freqs=num_freqs,
            num_frames=num_frames,
            freq_pad=freq_pad,
            time_pad=time_pad,
            patch_freq=p_f,
            patch_time=p_t,
            grid_freq=freq_pad // p_f,
            grid_time=time_pad // p_t,
            channels=cfg.model.in_channels,
        )

    def describe(self) -> str:
        return (
            f"F={self.num_freqs}->{self.freq_pad} T={self.num_frames}->{self.time_pad} "
            f"patch=({self.patch_freq},{self.patch_time}) grid=({self.grid_freq},{self.grid_time}) "
            f"tokens={self.num_patches} patch_dim={self.patch_dim}"
        )


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------
def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj)!r} into the config hash")


def _build(cls: type, data: Any, path: str) -> Any:
    if dataclasses.is_dataclass(cls):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected a mapping for {cls.__name__}, got {type(data)}")
        field_map = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(field_map)
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)} for {cls.__name__}")
        kwargs = {}
        for name, f in field_map.items():
            if name in data:
                kwargs[name] = _build(f.type, data[name], f"{path}.{name}" if path else name)
        return cls(**kwargs)

    origin = get_origin(cls)
    if origin is None:
        if cls is float and isinstance(data, int) and not isinstance(data, bool):
            return float(data)
        return data
    args = get_args(cls)
    if origin is list:
        return [_build(args[0], v, f"{path}[{i}]") for i, v in enumerate(data or [])]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], v, f"{path}[{i}]") for i, v in enumerate(data or ()))
        return tuple(
            _build(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, data, strict=False))
        )
    # Optional[...] / unions
    non_none = [a for a in args if a is not type(None)]
    if data is None:
        return None
    return _build(non_none[0], data, path)


def _resolve_types(cls: type) -> None:
    """Resolve ``from __future__ import annotations`` string hints once."""
    import typing

    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if isinstance(f.type, str):
            f.type = hints[f.name]


_TYPES_RESOLVED = False


def _ensure_types_resolved() -> None:
    global _TYPES_RESOLVED
    if _TYPES_RESOLVED:
        return
    for cls in (
        AudioConfig,
        STFTConfig,
        MelFilterConfig,
        ImageConfig,
        ModelConfig,
        FlowConfig,
        DatasetSpec,
        DataConfig,
        StatsConfig,
        OptimConfig,
        TrainConfig,
        SampleConfig,
        EvalConfig,
        TextConditionConfig,
        Config,
    ):
        _resolve_types(cls)
    _TYPES_RESOLVED = True


def config_from_dict(data: dict[str, Any]) -> Config:
    _ensure_types_resolved()
    cfg: Config = _build(Config, data, "")
    return cfg.validate()


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override {override!r} must look like a.b=value")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {override!r} traverses a non-mapping at {part!r}")
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """The one and only way to obtain a :class:`Config`."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    for override in overrides or []:
        _apply_override(data, override)
    return config_from_dict(data)
