import os
import tempfile
import docker
from pathlib import Path

from coding.constants import COMPETITION_ID
from ..helpers.git import GitRepo


def build_docker_container(logic_files: dict, hotkey: str) -> str:
    """
    Builds a Docker container for evaluating model logic.

    Args:
        logic_files (dict): Dictionary mapping filenames to file contents
        hotkey (str): Unique identifier for the logic

    Returns:
        str: ID of the built container
    """
    # Initialize Docker client
    client = docker.from_env()

    # Create temporary directory to store files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write logic files to temp directory
        for filename, content in logic_files.items():
            file_path = os.path.join(temp_dir, filename)
            # Create all parent directories
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file and write content
            with open(file_path, "w", encoding="latin-1") as f:
                f.write(content)
            print(f"Wrote {file_path}")
        # Copy Dockerfile and server files
        swe_server_path = Path(__file__).parent / "swe-server"
        for item in swe_server_path.glob("*"):
            if item.is_file():
                dest_path = os.path.join(temp_dir, item.name)
                with open(item, "rb") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
            elif item.is_dir():
                dest_dir = os.path.join(temp_dir, item.name)
                os.system(f"cp -r {item} {dest_dir}")

        # Build the container
        try:
            image, logs = client.images.build(
                path=temp_dir, tag=f"swe-logic-{str(hotkey)}-{COMPETITION_ID}", rm=True
            )
            return image.id

        except docker.errors.BuildError as e:
            print(f"Error building container: {str(e)}")
            raise
        except docker.errors.APIError as e:
            print(f"Docker API error: {str(e)}")
            raise


def run_docker_container(
    image_id: str, repo: GitRepo, hotkey: str
) -> docker.models.containers.Container:
    """
    Runs a Docker container for evaluating model logic.

    Args:
        image_id (str): ID of the Docker image to run
        repo (GitRepo): Git repository object containing code to evaluate
        hotkey (str): Unique identifier for the logic

    Returns:
        str: Name of the running container
    """
    # Initialize Docker client
    client = docker.from_env()

    container_name = f"swe-logic-{str(hotkey)}-{COMPETITION_ID}"

    try:
        # Remove any existing container with the same name
        try:
            existing = client.containers.get(container_name)
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        container = client.containers.create(
            image=image_id,
            name=container_name,
            detach=True,
            ports={"3000/tcp": 3000},
            extra_hosts={"host.docker.internal": "host-gateway"},
            volumes={repo.path: {"bind": "/app/repo", "mode": "ro"}},
            environment={"HOST_IP": os.getenv("HOST_IP", "localhost")}
        )
        # Start the container after creation
        container.start()
        return container
    except docker.errors.APIError as e:
        print(f"Docker API error: {str(e)}")
        raise