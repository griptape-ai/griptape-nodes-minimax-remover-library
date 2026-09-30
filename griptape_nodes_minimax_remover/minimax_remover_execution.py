"""Execution-time work for MiniMax-Remover: everything that reaches torch or diffusers.

No node module may import this at module scope. The orchestrator imports node modules to build
node classes, and it does not have the execution dependencies installed.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

logger = logging.getLogger(__name__)


def build_pipeline(repo_id: str, revision: str | None, device: str) -> Any:
    """Build the MiniMax-Remover pipeline on the given device.

    Downloads model weights from HuggingFace, imports custom modules from
    git submodule, and builds the pipeline.
    """
    dtype = torch.float16

    try:
        # The engine puts the library directory on sys.path, but not the submodule inside it.
        add_submodule_to_sys_path()

        # Import custom modules from submodule
        from transformer_minimax_remover import Transformer3DModel
        from pipeline_minimax_remover import Minimax_Remover_Pipeline

        # Load model components from HuggingFace
        # diffusers from_pretrained() handles downloading and caching automatically
        logger.info(f"Loading models from {repo_id} (revision: {revision})...")

        logger.info("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            repo_id,
            subfolder="vae",
            revision=revision,
            torch_dtype=dtype,
        )

        logger.info("Loading Transformer...")
        transformer = Transformer3DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            revision=revision,
            torch_dtype=dtype,
        )

        logger.info("Loading Scheduler...")
        scheduler = UniPCMultistepScheduler.from_pretrained(
            repo_id,
            subfolder="scheduler",
            revision=revision,
        )

        # Build pipeline
        logger.info("Building MiniMax-Remover pipeline...")
        pipeline = Minimax_Remover_Pipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        # Move to device
        logger.info(f"Moving pipeline to device: {device}")
        pipeline = pipeline.to(device)

        logger.info("Pipeline loaded successfully!")
        return pipeline

    except Exception as e:
        error_msg = f"Failed to build MiniMax-Remover pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def add_submodule_to_sys_path() -> None:
    """Make the custom MiniMax-Remover modules importable.

    The submodule ships flat modules (transformer_minimax_remover.py,
    pipeline_minimax_remover.py) rather than a package, so its own directory has to be on
    sys.path for them to resolve.
    """
    minimax_repo_path = str(Path(__file__).parent / "_minimax_remover_repo")
    if minimax_repo_path not in sys.path:
        sys.path.insert(0, minimax_repo_path)
        logger.debug(f"Added {minimax_repo_path} to sys.path")
