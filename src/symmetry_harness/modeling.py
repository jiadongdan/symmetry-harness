"""Legacy modeling entry points retained for explicit migration errors."""

from __future__ import annotations

def _moved(*args, **kwargs):
    raise RuntimeError(
        "Model construction, checkpoint loading, fine-tuning, and inference moved "
        "to the symmetry-learn Provider. Use the public symmetry-harness workflow."
    )


resolve_device = _moved
set_deterministic_seed = _moved
build_pretrained_model = _moved
load_pretrained_checkpoint = _moved
add_task_adapters = _moved
train_adapters = _moved
trainable_state_dict = _moved
save_adapter_checkpoint = _moved
predict_probabilities = _moved
