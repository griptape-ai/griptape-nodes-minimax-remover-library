"""MiniMax-Remover Library Advanced - Checks out the vendored MiniMax-Remover source"""

import logging
import subprocess
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimax_remover_library")


class MinimaxRemoverLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for MiniMax-Remover (AI-powered video object removal)."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library.

        This method handles submodule initialization. The pip packages are the engine's job:
        it installs them from the manifest, and the execution set lands in an environment this
        process cannot import from, so probing for them here would only ever report them missing.
        """
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # The submodule checkout below populates the execution environment, which only
        # the worker imports from. The orchestrator never reaches it, so skip here.
        if not GriptapeNodes.LibraryManager().is_worker:
            return

        try:
            self._init_minimax_remover_submodule()
            logger.info("MiniMax-Remover submodule initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize MiniMax-Remover: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory (where .venv lives)."""
        return Path(__file__).parent

    def _init_minimax_remover_submodule(self) -> Path:
        """Initialize the MiniMax-Remover git submodule."""
        library_root = self._get_library_root()
        minimax_submodule_dir = library_root / "_minimax_remover_repo"

        # Check if submodule is already initialized (has contents)
        if minimax_submodule_dir.exists() and any(minimax_submodule_dir.iterdir()):
            return minimax_submodule_dir

        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        git_repo_root = library_root.parent
        subprocess.check_call(["git", "-C", str(git_repo_root), "submodule", "update", "--init", "--recursive"])

        # Verify submodule was initialized
        if not minimax_submodule_dir.exists() or not any(minimax_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {minimax_submodule_dir} is empty or does not exist"
            )

        return minimax_submodule_dir
