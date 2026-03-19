"""MiniMax-Remover Library Advanced - Handles installation and setup for MiniMax-Remover dependencies"""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
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

        This method handles dependency installation and submodule initialization.
        """
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Check if all dependencies are properly installed
        if self._check_dependencies_installed():
            logger.info("All MiniMax-Remover dependencies are already installed, skipping installation")
            return

        logger.info("MiniMax-Remover dependencies or submodule not found, beginning installation process...")

        # Install dependencies and initialize submodule
        self._install_minimax_dependencies()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _check_dependencies_installed(self) -> bool:
        """Check if core dependencies are installed."""
        try:
            # Check torch (the main dependency that often fails)
            torch_version = version("torch")
            logger.debug(f"Found torch {torch_version}")

            # Check diffusers
            diffusers_version = version("diffusers")
            logger.debug(f"Found diffusers {diffusers_version}")

            # Check other key dependencies
            try:
                import decord
                logger.debug(f"Found decord")
            except ImportError:
                logger.debug("decord not found")
                return False

            return True

        except PackageNotFoundError as e:
            logger.debug(f"Dependency not found: {e}")
            return False

    def _install_minimax_dependencies(self) -> None:
        """Initialize MiniMax-Remover submodule.

        Note: Dependencies like torch are installed via UV using the JSON file.
        This method only handles submodule initialization.
        """
        try:
            logger.info("=" * 80)
            logger.info("Initializing MiniMax-Remover Library...")
            logger.info("=" * 80)

            # Initialize MiniMax-Remover submodule
            logger.info("Initializing MiniMax-Remover submodule...")
            self._init_minimax_remover_submodule()

            logger.info("MiniMax-Remover initialization completed successfully!")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Failed to initialize MiniMax-Remover: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

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
            return minimax_submodule_dir

        # Initialize submodule using pygit2 (recursive)
        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        # Verify submodule was initialized
        if not minimax_submodule_dir.exists() or not any(minimax_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {minimax_submodule_dir} is empty or does not exist"
            )

        return minimax_submodule_dir
