"""
Smart AI Lawn Mower - Webots simulation launcher.

This module starts the Webots simulation with the project world.
The LawnMower PROTO is responsible for starting its configured
Webots controller.
"""

import os
import subprocess
import sys
from pathlib import Path


def find_webots() -> Path:
    """
    Locate the Webots installation.

    Returns:
        Path to the Webots executable.

    Raises:
        FileNotFoundError:
            If a valid Webots installation cannot be found.
    """

    webots_home_env = os.environ.get("WEBOTS_HOME")

    if webots_home_env:
        webots_home = Path(webots_home_env)
    elif sys.platform == "win32":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "Webots",
            Path("C:/Program Files/Webots"),
        ]

        webots_home = next(
            (
                path
                for path in candidates
                if path.exists()
            ),
            None,
        )

        if webots_home is None:
            raise FileNotFoundError(
                "Webots installation could not be found."
            )

    elif sys.platform == "darwin":
        webots_home = Path("/Applications/Webots.app")

    else:
        candidates = [
            Path("/usr/local/webots"),
            Path("/opt/webots"),
        ]

        webots_home = next(
            (
                path
                for path in candidates
                if path.exists()
            ),
            None,
        )

        if webots_home is None:
            raise FileNotFoundError(
                "Webots installation could not be found."
            )

    if not webots_home.exists():
        raise FileNotFoundError(
            f"Webots installation not found at: {webots_home}"
        )

    # ------------------------------------------------------------------
    # Determine Webots executable
    # ------------------------------------------------------------------

    if sys.platform == "win32":
        executable = webots_home / "msys64" / "mingw64" / "bin" / "webots.exe"

        if not executable.exists():
            executable = webots_home / "webots.exe"

    elif sys.platform == "darwin":
        executable = (
            webots_home
            / "Contents"
            / "MacOS"
            / "webots"
        )

    else:
        executable = webots_home / "webots"

    if not executable.exists():
        raise FileNotFoundError(
            f"Webots executable not found at: {executable}"
        )

    return executable


def run_simulation() -> None:
    """
    Start the Webots lawn mower simulation.
    """

    project_root = Path(__file__).resolve().parent

    world_path = (
        project_root
        / "simulation"
        / "worlds"
        / "lawn_world.wbt"
    )

    if not world_path.exists():
        print(
            f"[ERROR] Webots world not found at: {world_path}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Locate Webots
    # ------------------------------------------------------------------

    try:
        webots_executable = find_webots()
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Launch information
    # ------------------------------------------------------------------

    print("==============================================")
    print("   SMART AI LAWN MOWER - SIMULATION LAUNCHER")
    print("==============================================")
    print(f"[Launcher] Webots: {webots_executable}")
    print(f"[Launcher] World:   {world_path}")
    print("==============================================")

    # ------------------------------------------------------------------
    # Webots environment
    # ------------------------------------------------------------------

    env = os.environ.copy()
    env["WEBOTS_HOME"] = str(webots_executable.parent)

    # ------------------------------------------------------------------
    # Start Webots
    #
    # The world file contains the LawnMower PROTO instance.
    # Webots will therefore start the controller specified by
    # the PROTO's controller field.
    # ------------------------------------------------------------------

    try:
        subprocess.run(
            [
                str(webots_executable),
                str(world_path),
            ],
            env=env,
            check=True,
        )

    except subprocess.CalledProcessError as error:
        print(
            "[ERROR] Webots exited with code "
            f"{error.returncode}"
        )
        sys.exit(error.returncode)

    except OSError as error:
        print(f"[ERROR] Failed to start Webots: {error}")
        sys.exit(1)


if __name__ == "__main__":
    run_simulation()