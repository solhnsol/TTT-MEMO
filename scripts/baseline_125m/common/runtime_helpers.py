"""Host/runtime helpers shared by sanity and experiments.

No TTT loss or gradient semantics. Lives in common/ so experiments do not
import sanity/.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    name: str
    compute_capability: str | None
    jax_device: str
    jax_backend: str


def describe_gpu() -> GpuInfo:
    import jax

    device = jax.devices()[0]
    name = getattr(device, "device_kind", str(device))
    compute_capability = None
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        smi_name, smi_cc = [part.strip() for part in line.split(",", 1)]
        name = smi_name
        compute_capability = smi_cc
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return GpuInfo(
        name=name,
        compute_capability=compute_capability,
        jax_device=str(device),
        jax_backend=jax.default_backend(),
    )


@dataclass
class AttentionOverride:
    """Process-local RTX 3060 compatibility workaround.

    Not a change to model architecture or attention semantics. The official
    prefix SWA path hardcodes jax.nn.dot_product_attention(..., implementation="cudnn").
    The wrapper only rewrites that keyword to "xla".
    """

    enabled: bool
    n_seen: int = 0
    n_redirected: int = 0
    n_passthrough: int = 0
    installed: bool = False

    def install(self) -> None:
        if not self.enabled:
            return
        import jax
        import jax.nn as jnn

        original = jnn.dot_product_attention

        def wrapped(*args, **kwargs):
            self.n_seen += 1
            if kwargs.get("implementation") == "cudnn":
                kwargs = dict(kwargs)
                kwargs["implementation"] = "xla"
                self.n_redirected += 1
            else:
                self.n_passthrough += 1
            return original(*args, **kwargs)

        jnn.dot_product_attention = wrapped
        jax.nn.dot_product_attention = wrapped
        self.installed = True

    def report_for_config(self, *, force_flash: bool, seq_modeling_block: str) -> dict:
        prefix_requested = "cudnn"
        suffix_requested = "cudnn" if force_flash else None
        prefix_effective = "xla" if self.enabled else prefix_requested
        suffix_effective = "xla" if (self.enabled and suffix_requested == "cudnn") else suffix_requested
        return {
            "override_enabled": self.enabled,
            "override_installed": self.installed,
            "seq_modeling_block": seq_modeling_block,
            "force_flash": force_flash,
            "prefix_requested_backend": prefix_requested,
            "prefix_effective_backend": prefix_effective,
            "suffix_requested_backend": suffix_requested,
            "suffix_effective_backend": suffix_effective,
            "redirects_only_when_implementation_is_cudnn": True,
            "n_dot_product_attention_seen": self.n_seen,
            "n_cudnn_redirected_to_xla": self.n_redirected,
            "n_passthrough": self.n_passthrough,
        }


def tokens_to_batch(tokens, *, bos_token_id: int):
    """Same slicing as `ttt.dataloader.lm_dataset._to_batch`, minus the unused eos arg."""
    from ttt.model.data import Batch

    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )
