"""MiniMax-Remover Video Object Removal Node"""

import gc
import logging
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video
from diffusers_nodes_library.common.utils.torch_utils import get_best_device
from griptape.artifacts import VideoUrlArtifact
from PIL import Image

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import (
    HuggingFaceRepoParameter,
)
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)


class MinimaxRemoverVideoNodeParameters:
    """Parameters for MiniMax-Remover Video Object Removal Node."""

    def __init__(self, node: BaseNode):
        """Initialize parameters for MiniMax-Remover video processing."""
        self._node = node

        # HuggingFace repository parameter for model weights
        self._huggingface_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=["zibojia/minimax-remover"],
        )

    def add_input_parameters(self) -> None:
        """Add all input parameters to the node."""
        # Add HuggingFace repo parameter (model weights)
        self._huggingface_repo_parameter.add_input_parameters()

        # Inference steps parameter
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                input_types=["int"],
                type="int",
                tooltip="Number of denoising steps (6-50). More steps = better quality but slower processing",
                default_value=12,
                traits={Slider(min_val=6, max_val=50)},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Height parameter (must be multiple of 8)
        self._node.add_parameter(
            Parameter(
                name="height",
                input_types=["int"],
                type="int",
                tooltip="Output video height in pixels (256-1024, must be multiple of 8)",
                default_value=480,
                traits={Slider(min_val=256, max_val=1024)},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"step": 8},
            )
        )

        # Width parameter (must be multiple of 8)
        self._node.add_parameter(
            Parameter(
                name="width",
                input_types=["int"],
                type="int",
                tooltip="Output video width in pixels (256-1024, must be multiple of 8)",
                default_value=832,
                traits={Slider(min_val=256, max_val=1024)},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"step": 8},
            )
        )

        # Number of frames parameter
        self._node.add_parameter(
            Parameter(
                name="num_frames",
                input_types=["int"],
                type="int",
                tooltip="Number of frames to process (1-81). Must match input video frame count",
                default_value=81,
                traits={Slider(min_val=1, max_val=81)},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

    def build_pipeline(self) -> Any:
        """Build the MiniMax-Remover pipeline.

        Downloads model weights from HuggingFace, imports custom modules from
        git submodule, and builds the pipeline.
        """
        repo_id, revision = self._huggingface_repo_parameter.get_repo_revision()

        logger.info("Building MiniMax-Remover pipeline...")
        logger.info(f"Repository: {repo_id}, Revision: {revision}")

        device = get_best_device()
        dtype = torch.float16

        try:
            # Import custom modules (node should have added submodule to sys.path)
            logger.info("Importing custom MiniMax-Remover modules...")
            try:
                from transformer_minimax_remover import Transformer3DModel
                from pipeline_minimax_remover import Minimax_Remover_Pipeline
            except ImportError as e:
                error_msg = (
                    f"Failed to import MiniMax-Remover custom modules: {e}. "
                    "Ensure the git submodule is initialized and added to sys.path."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

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


class MinimaxRemoverVideoNode(ControlNode):
    """Node for removing objects from videos using MiniMax-Remover.

    This node uses AI-powered diffusion models to seamlessly remove unwanted
    objects from videos. Provide a source video and a mask video (white=remove,
    black=keep) to remove objects while maintaining temporal consistency.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the MiniMax-Remover video node."""
        super().__init__(**kwargs)

        # Ensure custom modules are available before creating parameters
        self._ensure_minimax_modules_available()

        # Initialize parameters
        self.params = MinimaxRemoverVideoNodeParameters(self)
        self.params.add_input_parameters()

        # Initialize logging parameter
        self.log_params = LogParameter(self)

        # Input video parameter
        self.add_parameter(
            Parameter(
                name="input_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Source video containing the object to remove",
            )
        )

        # Input mask parameter
        self.add_parameter(
            Parameter(
                name="input_mask",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Binary mask video (white pixels = remove, black pixels = keep). Must match input video frame count.",
            )
        )

        # Output video parameter
        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Processed video with object removed",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Add logs output parameter
        self.log_params.add_output_parameters()

    def _ensure_minimax_modules_available(self):
        """Add _minimax_remover_repo to sys.path for lazy imports.

        This allows importing the custom MiniMax-Remover modules
        (transformer_minimax_remover.py and pipeline_minimax_remover.py)
        from the git submodule.
        """
        minimax_repo_path = str(Path(__file__).parent / "_minimax_remover_repo")

        if minimax_repo_path not in sys.path:
            sys.path.insert(0, minimax_repo_path)
            logger.debug(f"Added {minimax_repo_path} to sys.path")

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate inputs before processing."""
        errors = []

        # Check that both inputs are provided
        if not self.get_parameter_value("input_video"):
            errors.append(Exception("Input video is required"))

        if not self.get_parameter_value("input_mask"):
            errors.append(Exception("Mask video is required"))

        # Validate height and width are multiples of 8
        height = self.get_parameter_value("height")
        width = self.get_parameter_value("width")

        if height is not None and height % 8 != 0:
            errors.append(Exception(f"Height must be a multiple of 8, got {height}"))

        if width is not None and width % 8 != 0:
            errors.append(Exception(f"Width must be a multiple of 8, got {width}"))

        return errors or None

    async def aprocess(self) -> None:
        """Process the video to remove objects using MiniMax-Remover."""
        await self._process()

    async def _process(self) -> None:
        """Internal processing implementation."""
        start_time = time.time()

        try:
            # Get parameters
            height = self.get_parameter_value("height")
            width = self.get_parameter_value("width")
            num_frames = self.get_parameter_value("num_frames")
            num_inference_steps = self.get_parameter_value("num_inference_steps")

            self.log_params.append_to_logs(f"Parameters: {width}x{height}, {num_frames} frames, {num_inference_steps} steps\n")

            # Load input video
            self.log_params.append_to_logs("Loading input video...\n")
            input_video_artifact = self.get_parameter_value("input_video")
            input_video_path = self._get_video_path(input_video_artifact)
            video_frames = load_video(input_video_path)
            self.log_params.append_to_logs(f"Loaded {len(video_frames)} frames from input video\n")

            # Load mask video
            self.log_params.append_to_logs("Loading mask video...\n")
            mask_video_artifact = self.get_parameter_value("input_mask")
            mask_video_path = self._get_video_path(mask_video_artifact)
            mask_frames = load_video(mask_video_path)
            self.log_params.append_to_logs(f"Loaded {len(mask_frames)} frames from mask video\n")

            # Validate frame counts match
            if len(video_frames) != len(mask_frames):
                raise ValueError(
                    f"Frame count mismatch: input video has {len(video_frames)} frames, "
                    f"mask video has {len(mask_frames)} frames. Both must have the same number of frames."
                )

            # Validate frame count matches parameter
            if len(video_frames) != num_frames:
                self.log_params.append_to_logs(
                    f"Warning: Input video has {len(video_frames)} frames, "
                    f"but num_frames parameter is set to {num_frames}. "
                    f"Using actual frame count: {len(video_frames)}\n"
                )
                num_frames = len(video_frames)

            # Build pipeline
            self.log_params.append_to_logs("Building MiniMax-Remover pipeline...\n")
            pipeline = self.params.build_pipeline()
            device = pipeline.device
            self.log_params.append_to_logs(f"Pipeline built on device: {device}\n")

            # Prepare frames in expected format [f, h, w, c]
            self.log_params.append_to_logs("Preparing video frames...\n")
            processed_video_frames = []
            processed_mask_frames = []

            for i, (video_frame, mask_frame) in enumerate(zip(video_frames, mask_frames)):
                # Convert to PIL if needed
                if not isinstance(video_frame, Image.Image):
                    video_frame = Image.fromarray(video_frame)
                if not isinstance(mask_frame, Image.Image):
                    mask_frame = Image.fromarray(mask_frame)

                # Resize frames
                video_frame = video_frame.resize((width, height), Image.Resampling.BILINEAR)
                mask_frame = mask_frame.resize((width, height), Image.Resampling.NEAREST)

                # Convert to numpy arrays
                video_np = np.array(video_frame).astype(np.float32) / 255.0  # [h, w, c] in range [0, 1]
                video_np = (video_np - 0.5) / 0.5  # Normalize to [-1, 1]

                mask_np = np.array(mask_frame.convert('L')).astype(np.float32) / 255.0  # [h, w] in range [0, 1]
                mask_np = mask_np[:, :, None]  # [h, w, 1]

                processed_video_frames.append(video_np)
                processed_mask_frames.append(mask_np)

            # Stack into [f, h, w, c] format
            images = np.stack(processed_video_frames, axis=0)  # [f, h, w, c]
            masks = np.stack(processed_mask_frames, axis=0)    # [f, h, w, 1]

            self.log_params.append_to_logs(f"Images shape: {images.shape}\n")
            self.log_params.append_to_logs(f"Masks shape: {masks.shape}\n")

            # Convert both to torch tensors (required by pipeline)
            images_tensor = torch.from_numpy(images).to(device)
            masks_tensor = torch.from_numpy(masks).to(device)
            self.log_params.append_to_logs(f"Converted to tensors - images: {images_tensor.shape}, masks: {masks_tensor.shape}\n")

            # Run inference
            self.log_params.append_to_logs(f"Running inference with {num_inference_steps} steps...\n")
            inference_start = time.time()

            with torch.inference_mode():
                output = pipeline(
                    images=images_tensor,
                    masks=masks_tensor,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                )

            inference_time = time.time() - inference_start
            self.log_params.append_to_logs(f"Inference completed in {inference_time:.2f}s\n")

            # Get output frames
            self.log_params.append_to_logs("Processing output frames...\n")
            # Pipeline returns [batch=1, frames, height, width, channels]
            # Extract first batch element: .frames[0] -> [frames, height, width, channels]
            result = output.frames[0]
            self.log_params.append_to_logs(f"Output shape: {result.shape}, dtype: {result.dtype}\n")

            # Denormalize from [-1, 1] to [0, 1] for export_to_video
            result = (result + 1.0) / 2.0
            result = np.clip(result, 0.0, 1.0)
            self.log_params.append_to_logs(f"After denormalization: shape={result.shape}, min={result.min():.3f}, max={result.max():.3f}\n")

            # Export to video file and publish
            self.log_params.append_to_logs("Exporting video...\n")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
                export_path = Path(temp_file_obj.name)

            try:
                # export_to_video handles numpy array [f, h, w, c] in range [0, 1]
                export_to_video(result, str(export_path), fps=16)

                # Publish to static files
                filename = f"{uuid.uuid4()}{export_path.suffix}"
                output_url = GriptapeNodes.StaticFilesManager().save_static_file(
                    export_path.read_bytes(),
                    filename
                )

                # Create output artifact
                output_artifact = VideoUrlArtifact(value=output_url)
            finally:
                if export_path.exists():
                    export_path.unlink()

            # Clean up pipeline and free memory
            self.log_params.append_to_logs("Cleaning up pipeline...\n")
            del pipeline
            del images_tensor
            del masks_tensor
            del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log_params.append_to_logs("Memory cleaned up\n")

            # Log completion
            total_time = time.time() - start_time
            self.log_params.append_to_logs(f"Processing completed in {total_time:.2f}s\n")
            self.log_params.append_to_logs(f"Output video: {output_url}\n")

            # Set output parameters
            self.set_parameter_value("output_video", output_artifact)

        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.log_params.append_to_logs(f"ERROR: {error_msg}\n")
            raise

    def _get_video_path(self, video_artifact: Union[VideoUrlArtifact, File]) -> str:
        """Get the local file path for a video artifact.

        Args:
            video_artifact: Video artifact (URL or File)

        Returns:
            str: Local file path to the video
        """
        if isinstance(video_artifact, File):
            return video_artifact.path

        elif isinstance(video_artifact, VideoUrlArtifact):
            # Download URL to temp file
            video_bytes = File(video_artifact.value).read_bytes()
            fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            import os
            os.close(fd)  # Close the file descriptor immediately
            try:
                Path(temp_path).write_bytes(video_bytes)
            except Exception:
                # Clean up on failure
                Path(temp_path).unlink(missing_ok=True)
                raise
            return temp_path

        else:
            raise TypeError(f"Unsupported video artifact type: {type(video_artifact)}")
