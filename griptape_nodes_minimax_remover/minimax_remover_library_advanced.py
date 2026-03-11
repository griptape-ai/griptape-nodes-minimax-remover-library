"""MiniMax-Remover Library Advanced - Handles installation and setup for MiniMax-Remover dependencies"""

import logging
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minimax_remover_library")


class MinimaxRemoverLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for MiniMax-Remover (AI-powered video object removal)."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library.

        This method handles the initialization of the MiniMax-Remover git submodule.
        """
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Check if submodule is already initialized
        if self._check_submodule_initialized():
            logger.info("MiniMax-Remover submodule already initialized, skipping initialization")
            return

        logger.info("MiniMax-Remover submodule not found, beginning initialization...")

        # Initialize submodule
        self._init_minimax_remover_submodule()

        logger.info("MiniMax-Remover submodule initialization completed successfully!")

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory (where .venv lives)."""
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        """Get the Python executable path from the library's venv.

        Returns the path to the venv's Python executable, which differs between
        Windows (Scripts/python.exe) and Unix (bin/python).
        """
        venv_path = self._get_library_root() / ".venv"

        if GriptapeNodes.OSManager().is_windows():
            venv_python_path = venv_path / "Scripts" / "python.exe"
        else:
            venv_python_path = venv_path / "bin" / "python"

        if not venv_python_path.exists():
            raise RuntimeError(
                f"Library venv Python not found at {venv_python_path}. "
                "The library venv must be initialized before loading."
            )

        logger.debug(f"Python executable found at: {venv_python_path}")
        return venv_python_path

    def _check_submodule_initialized(self) -> bool:
        """Check if the MiniMax-Remover submodule is initialized (has contents)."""
        library_root = self._get_library_root()
        minimax_submodule_dir = library_root / "_minimax_remover_repo"

        # Check if submodule directory exists and has contents
        if minimax_submodule_dir.exists() and any(minimax_submodule_dir.iterdir()):
            # Verify required files exist
            required_files = [
                minimax_submodule_dir / "pipeline_minimax_remover.py",
                minimax_submodule_dir / "transformer_minimax_remover.py"
            ]

            if all(f.exists() for f in required_files):
                logger.debug(f"MiniMax-Remover submodule found at {minimax_submodule_dir}")
                return True
            else:
                logger.warning(f"Submodule directory exists but missing required files")
                return False

        logger.debug("MiniMax-Remover submodule not initialized")
        return False

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules.

        Pygit2 does not have a built-in recursive update.
        Equivalent to: git submodule update --init --recursive
        """
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        # Recursively update nested submodules
        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_minimax_remover_submodule(self) -> Path:
        """Initialize the MiniMax-Remover git submodule."""
        library_root = self._get_library_root()
        minimax_submodule_dir = library_root / "_minimax_remover_repo"

        # Check if submodule is already initialized (has contents)
        if minimax_submodule_dir.exists() and any(minimax_submodule_dir.iterdir()):
            logger.info(f"Submodule already exists at {minimax_submodule_dir}")
            return minimax_submodule_dir

        # Initialize submodule using pygit2 (recursive)
        logger.info("Initializing MiniMax-Remover submodule using pygit2...")
        git_repo_root = library_root.parent

        try:
            self._update_submodules_recursive(git_repo_root)
        except Exception as e:
            error_msg = f"Failed to initialize submodule: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # Verify submodule was initialized
        if not minimax_submodule_dir.exists() or not any(minimax_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {minimax_submodule_dir} is empty or does not exist. "
                "Please ensure .gitmodules is configured correctly and git is available."
            )

        # Verify required files exist
        required_files = [
            minimax_submodule_dir / "pipeline_minimax_remover.py",
            minimax_submodule_dir / "transformer_minimax_remover.py"
        ]

        missing_files = [f.name for f in required_files if not f.exists()]
        if missing_files:
            raise RuntimeError(
                f"Submodule initialized but missing required files: {', '.join(missing_files)}"
            )

        logger.info(f"Submodule successfully initialized at {minimax_submodule_dir}")
        return minimax_submodule_dir
