"""
Weight converter: load weights from original CausalWanModel into TRTCausalWanModel.

The two models share the same linear layers, convolutions, and norms — they just
have slightly different module paths due to the TRT-safe wrappers. This utility
handles the mapping.
"""

import torch
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


def map_original_to_trt(original_key: str) -> Optional[str]:
    """
    Map a parameter name from CausalWanModel to TRTCausalWanModel.
    
    The mapping is mostly 1:1 since we preserved layer names. The main
    differences are:
    1. Original has CausalWanSelfAttention -> TRT has TRTSelfAttention
    2. Original cross_attn is WanT2VCrossAttention -> TRT has TRTCrossAttention
    3. Original CausalHead -> TRTCausalHead
    
    But since we kept the same attribute names (q, k, v, o, norm_q, norm_k, etc.)
    the parameter paths are identical.
    
    Returns None for parameters that should be skipped (e.g. freqs which
    we precompute differently).
    """
    # Skip the complex RoPE frequencies — TRT model uses real sin/cos buffers
    if 'freqs' in original_key and 'rope_' not in original_key:
        # This is the original complex freqs buffer
        return None
    
    # Everything else maps directly
    return original_key


def convert_original_state_dict(
    original_state_dict: dict,
    trt_model: torch.nn.Module,
) -> OrderedDict:
    """
    Convert an original CausalWanModel state dict to TRTCausalWanModel format.
    
    Args:
        original_state_dict: State dict from CausalWanModel checkpoint
        trt_model: Target TRTCausalWanModel instance (for shape validation)
        
    Returns:
        Converted state dict ready for trt_model.load_state_dict()
    """
    trt_state = OrderedDict()
    trt_keys = set(trt_model.state_dict().keys())
    original_keys = set(original_state_dict.keys())
    
    mapped = 0
    skipped = 0
    
    for orig_key, param in original_state_dict.items():
        trt_key = map_original_to_trt(orig_key)
        
        if trt_key is None:
            logger.debug(f"Skipping: {orig_key}")
            skipped += 1
            continue
        
        if trt_key in trt_keys:
            # Shape validation
            expected_shape = trt_model.state_dict()[trt_key].shape
            if param.shape != expected_shape:
                logger.warning(
                    f"Shape mismatch for {trt_key}: "
                    f"original={param.shape}, expected={expected_shape}"
                )
                continue

            trt_state[trt_key] = param
            mapped += 1
        else:
            logger.debug(f"No TRT match for: {orig_key}")
            skipped += 1
    
    # Check for missing keys in TRT model
    missing = trt_keys - set(trt_state.keys())
    # Filter out buffers (rope_cos/sin) which are pre-computed
    missing_params = {k for k in missing if 'rope_' not in k}
    
    if missing_params:
        logger.warning(f"Missing parameters in converted state dict: {missing_params}")
    
    logger.info(
        f"Weight conversion: {mapped} mapped, {skipped} skipped, "
        f"{len(missing)} missing (including {len(missing) - len(missing_params)} RoPE buffers)"
    )
    
    return trt_state


def load_checkpoint_for_trt(
    checkpoint_path: str,
    trt_model: torch.nn.Module,
    strict: bool = False,
) -> None:
    """
    Load a CausalWanModel checkpoint into a TRTCausalWanModel.
    
    Args:
        checkpoint_path: Path to model.pt checkpoint
        trt_model: TRTCausalWanModel instance
        strict: Whether to require exact key matching
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict
    if isinstance(ckpt, dict):
        if 'generator' in ckpt:
            state_dict = ckpt['generator']
        elif 'generator_ema' in ckpt:
            state_dict = ckpt['generator_ema']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    
    # The checkpoint state dict may have a prefix like 'model.' from the wrapper
    # Check and strip if needed
    prefix_to_strip = None
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('model.'):
        prefix_to_strip = 'model.'
    
    if prefix_to_strip:
        logger.info(f"Stripping prefix '{prefix_to_strip}' from state dict keys")
        state_dict = {
            k[len(prefix_to_strip):]: v for k, v in state_dict.items()
            if k.startswith(prefix_to_strip)
        }
    
    # Convert
    trt_state = convert_original_state_dict(state_dict, trt_model)
    
    # Load
    missing, unexpected = trt_model.load_state_dict(trt_state, strict=strict)
    
    if missing:
        # Filter out expected missing (RoPE buffers)
        missing_real = [k for k in missing if 'rope_' not in k]
        if missing_real:
            logger.warning(f"Missing keys after load: {missing_real}")
        else:
            logger.info("All missing keys are expected RoPE buffers")
    
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    logger.info("Checkpoint loaded into TRT model successfully")
