import os
import time
import docker
import shutil
import difflib
import tempfile
import logging
import threading
import traceback
import bittensor as bt
from docker import DockerClient
from pathlib import Path, PurePosixPath
from typing import Callable, List, Dict

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    KEY_PREDICTION,
    LOG_TEST_OUTPUT,
    UTF8,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    copy_to_container,
)
from swebench.harness.docker_build import (
    BuildImageError,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.utils import (
    EvaluationError,
)

from .task import Task
from coding.helpers.git import GitRepo
from coding.constants import IMAGE_VERSION
from coding.helpers.containers import DockerServer
from coding.finetune.dockerutil import exec_run_with_timeout
from coding.schemas import Context, Patch, ChangedFile, ChangedFiles, apply_edits

def normalize_image_name(image_name):
    import re
    # Find all IP:port patterns
    ip_port_pattern = r'\d+\.\d+\.\d+\.\d+:\d+'
    matches = re.findall(ip_port_pattern, image_name)
    
    # If there are multiple IP:port matches, keep only the first one
    if len(matches) > 1:
        for match in matches[1:]:
            image_name = image_name.replace(match + '/', '')
            image_name = image_name.replace(match, '')
    
    return image_name

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=8 -p1 -l",
]


def run_instance(
    repo: GitRepo,
    instance: dict,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    image_name: str = None,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    test_spec = make_test_spec(
        instance, namespace="swebench", instance_image_tag="latest"
    )
    # Set up logging directory
    instance_id = test_spec.instance_id
    logger = logging.getLogger()
    with tempfile.NamedTemporaryFile(delete=False) as temp_log_file:
        setattr(logger, "log_file", temp_log_file.name)
    # Run the instance
    container = None
    try:
        print(f"Creating container for {instance_id} from image {image_name}...")
        # Check if container with this name already exists and remove it
        try:
            existing_container = client.containers.get(instance_id)
            print(f"Container {instance_id} already exists, removing it...")
            existing_container.remove(force=True)
        except docker.errors.NotFound:
            pass  # Container doesn't exist, which is fine
        # Create and start instance container from the existing image
        container = client.containers.create(
            image=image_name, name=instance_id, command="tail -f /dev/null"
        )
        container.start()
        print(f"Container for {instance_id} created and started: {container.id}")
        # Delete everything in /testbed/ and copy new files
        # container.exec_run("rm -rf /testbed/*")
        # os.system(f"docker cp {repo.temp_dir}/. {instance_id}:/testbed/")
        container.exec_run(
            f"git reset --hard {instance['base_commit']}", workdir="/testbed"
        )
        container.exec_run(
            "git config --global --add safe.directory /testbed", workdir="/testbed"
        )

        

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as log_dir:
            log_dir = Path(log_dir)
            # Copy model prediction as patch file to container
            patch_file = log_dir / "patch.diff"
            # pred[KEY_PREDICTION] = pred[KEY_PREDICTION]
            patch_file.write_text(pred[KEY_PREDICTION] or "")
            # print(
            # f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
            # )
            copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))
            # print("THE PATCH is: ", pred[KEY_PREDICTION].split("\n"))
            # print(container.exec_run("ls", workdir="/testbed", user="root").output.decode(UTF8))
            # Attempt to apply patch to container (TODO: FIX THIS)
            applied_patch = False
            for git_apply_cmd in GIT_APPLY_CMDS:
                val = container.exec_run(
                    f"{git_apply_cmd} {DOCKER_PATCH}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                if val.exit_code == 0:
                    # print(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                    applied_patch = True
                    break
                # else:
                    # print(f"Failed to apply patch to container: {git_apply_cmd}")
                    # print("The error is: ", val.output.decode(UTF8))
                    # print("The patch is: ", pred[KEY_PREDICTION])

            if not applied_patch:
                print(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
                raise EvaluationError(
                    instance_id,
                    f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                    logger,
                )
            # Get git diff before running eval script
            git_diff_output_before = (
                container.exec_run(
                    "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
                )
                .output.decode(UTF8)
                .strip()
            )

            eval_file = Path(log_dir / "eval.sh")
            eval_file.write_text(test_spec.eval_script)
            # print(
            #     f"Eval script for {instance_id} written to {eval_file}; copying to container..."
            # )
            copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))
            # Run eval script, write output to logs
            test_output, timed_out, total_runtime = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", timeout
            )
            test_output_path = log_dir / LOG_TEST_OUTPUT
            print(f"Test runtime: {total_runtime:_.2f} seconds")
            with open(test_output_path, "w") as f:
                f.write(test_output)
                # print(f"Test output for {instance_id} written to {test_output_path}")
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                        logger,
                    )

            # Get git diff after running eval script (ignore permission changes)
            git_diff_output_after = (
                container.exec_run(
                    "git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR
                )
                .output.decode(UTF8)
                .strip()
            )

            # Get report from test output
            print(f"Grading answer for {instance_id}...")
            report = get_eval_report(
                test_spec=test_spec,
                prediction=pred,
                test_log_path=test_output_path,
                include_tests_status=True,
            )
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
        )
        print(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)

    return


def score_patch(
    patch: str, repo: GitRepo, instance: dict, client: docker.DockerClient, image_name: str
):
    # if patch.strip() == "":
        # return 0

    prediction = {
        "instance_id": instance["instance_id"],
        "model_patch": patch,
        "raw_model_patch": patch,
        "model_name_or_path": "gpt-4o",
        "original_file_content": "",
    }
    try:
        result = run_instance(
            repo, instance, prediction, False, False, client, "nil", 600, image_name
        )
        if result[1][instance["instance_id"]]["resolved"]:
            return 1
        else:
            return 0
    except Exception as e:
        print("There was an error scoring the patch: ", e)
        print(traceback.format_exc())
        return 0


def add_newlines(lines: list[str]) -> list[str]:
    """
    Adds a \n character to each line except the last
    """
    with_newlines = [line + "\n" for line in lines[:-1]]
    if len(lines) > 0 and len(lines[-1]) > 0:
        with_newlines.append(lines[-1])  # Append the last line without a newline
    return with_newlines


def create_diff(changes: list[ChangedFile]) -> str:
    all_hunks = []

    for change in changes:
        before_lines = add_newlines([line for line in change.old_content.split("\n")])
        after_lines = add_newlines([line for line in change.new_content.split("\n")])
        # fix bug where the last line is the same but theres a whitespace difference
        if (
            len(before_lines) > 0
            and len(after_lines) > 0
            and before_lines[-1].strip() == after_lines[-1].strip()
        ):
            after_lines[-1] = before_lines[-1]
        from_file = f"a/{change.file_name}"
        to_file = f"b/{change.file_name}"

        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=from_file,
            tofile=to_file,
            lineterm="\n",
            n=3,  # Number of context lines
        )

        hunk = "".join(diff)

        if hunk:  # Only add non-empty hunks
            all_hunks.append(f"diff --git {from_file} {to_file}")
            all_hunks.append(hunk)

    return "\n".join(all_hunks)


def grab_file_from_repo(repo_path: str, file_path: str) -> str:
    with open(os.path.join(repo_path, file_path), "r") as f:
        return f.read()

def git_diff_to_changed_files(diff: str, repo_path: str) -> ChangedFiles:
    # This function was generated using Cursor and not verified. It is only used to get the changed files to later be used to find the file names, the old and new content may be incorrect.
    changed_files = []
    file_name = None
    
    lines = diff.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Start of a new file diff
        if line.startswith("diff --git"):
            # Save the previous file if there was one
            if file_name:
                changed_files.append(
                    ChangedFile(
                        file_name=file_name,
                        old_content="",
                        new_content=""
                    )
                )
            
            # Reset for new file
            parts = line.split()
            if len(parts) >= 4:
                file_name = parts[3][2:]  # Extract from b/path/to/file
        i += 1
    
    # Add the last file if there is one
    if file_name:
        changed_files.append(
            ChangedFile(
                file_name=file_name,
                old_content="",
                new_content=""
            )
        )
    
    return ChangedFiles(files=[file.model_dump() for file in changed_files])

def patch_to_changed_files(patch: Patch | str, repo_path: str) -> ChangedFiles:
    changed_files = []
    file_edits = {}
    for edit in patch.edits:
        file_path = edit.file_name
        old_content = grab_file_from_repo(repo_path, file_path)
        if file_path not in file_edits:
            file_edits[file_path] = []
        file_edits[file_path].append(edit)
    for file_path, edits in file_edits.items():
        old_content = grab_file_from_repo(repo_path, file_path)
        new_content = apply_edits(old_content, edits)
        changed_files.append(
            ChangedFile(
                file_name=file_path, old_content=old_content, new_content=new_content
            )
        )
    return ChangedFiles(files=[file.model_dump() for file in changed_files])


class SWEBenchTask(Task):
    name: str = "swebench"
    desc: str = "given a github issue corrrectly solve it"
    goal: str = "return the valid patch"
    reward_definition: str = []
    penalty_definition: List = []
    cleaning_pipeline: List = []  # TODO remove markdown wrappings
    dataset_options: Dict = {}
    attachments = []
    messages = []
    files = []

    def __init__(
        self,
        llm: Callable,
        context: Context,
        docker_server=None,
        use_remote: bool = False,
    ):
        self.repo = GitRepo(context.title, context.extras["base_commit"])
        self.row = context.extras["row"]
        self.use_remote = use_remote
        if docker_server is None:
            self.docker_server = DockerServer(
                remote_host_url=os.getenv("REMOTE_DOCKER_HOST", None),
                remote_host_registry=f"{os.getenv('DOCKER_HOST_IP', None)}:5000",
            )
        else:
            self.docker_server = docker_server
        self.image_name = f"swe-eval-{self.row['repo']}-{self.row['version']}:{IMAGE_VERSION}"
        if (
            self.use_remote
            and hasattr(self.docker_server, "remote")
            and self.docker_server.remote
        ):
            docker_host_ip = os.getenv("DOCKER_HOST_IP")
            normalized_name = normalize_image_name(self.image_name)
            self.image_name = f"{docker_host_ip}:5000/{normalized_name}"
        self._build_image()

        self.context = context
        self.query = context.topic
        self.base_commit = context.extras["base_commit"]
        self.pull_number = context.extras["pull_number"]
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def _build_image(self):
        test_spec = make_test_spec(
            self.row, namespace="swebench", instance_image_tag="latest"
        )

        # Check if image already exists
        client = (
            self.docker_server._local_client
            if not self.use_remote or not self.docker_server.remote
            else self.docker_server._remote_client
        )

        try:
            client.images.get(self.image_name)
            print(f"Image {self.image_name} already exists, skipping build")
            return
        except:
            print(f"Building image {self.image_name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_script = test_spec.install_repo_script.replace("reset --hard", "checkout -f")
            with open(os.path.join(temp_dir, "setup_repo.sh"), "w") as f:
                f.write(repo_script)
            dockerfile_content = f"""
FROM "brokespace/swe-env-{test_spec.repo.replace("/", "-")}-{test_spec.version}:{IMAGE_VERSION}"
USER root

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
"""
            with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
                f.write(dockerfile_content)
            start_time = time.time()
            for _ in range(3): # try 3 times
                try:
                    if (
                        self.use_remote
                        and hasattr(self.docker_server, "remote")
                        and self.docker_server.remote
                    ):
                        self.docker_server.remote.build(
                            path=temp_dir, tag=self.image_name, push=False
                        )
                    else:
                        self.docker_server.local.build(path=temp_dir, tag=self.image_name)
                    break
                except Exception as e:
                    print("There was an error building the image: ", e)
                    print(traceback.format_exc())
            if self.use_remote and hasattr(self.docker_server, "remote") and self.docker_server.remote:
                self.docker_server._remote_client.images.pull(self.image_name)
            end_time = time.time()
            build_duration = end_time - start_time
            print(f"Building the Docker image took {build_duration:.2f} seconds.")

    def __getstate__(self):
        # Remove the Docker image before pickling
        # self.client.images.remove(image=self.image_name, force=True)
        state = self.__dict__.copy()
        state["docker_server"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild the Docker image after unpickling
        self.docker_server = DockerServer(
            remote_host_url=os.getenv("REMOTE_DOCKER_HOST", None),
            remote_host_registry=f"{os.getenv('DOCKER_HOST_IP', None)}:5000",
        )
        if (
            self.use_remote
            and hasattr(self.docker_server, "remote")
            and self.docker_server.remote
            and os.getenv("DOCKER_HOST_IP") not in self.image_name
        ):
            docker_host_ip = os.getenv("DOCKER_HOST_IP")
            self.image_name = f"{docker_host_ip}:5000/{self.image_name}"
        # Extract the version part from the image name, handling multiple colons
        image_parts = self.image_name.split(":")
        if len(image_parts) > 1 and image_parts[-1] != IMAGE_VERSION:
            self.image_name = ":".join(image_parts[:-1]) + ":" + IMAGE_VERSION
        
        self._build_image()

    # def __del__(self):
    #     # Ensure the Docker image is removed when the object is deleted
    #     try:
    #         self.client.images.remove(image=self.image_name, force=True)
    #     except Exception as e:
    #         bt.logging.warning(f"Failed to remove Docker image: {e}")

    def score(self, patch: Patch | str):
        try:
            changed_files = patch_to_changed_files(patch, self.repo.path) if isinstance(patch, Patch) else git_diff_to_changed_files(patch, self.repo.path)
            if any("test" in file.file_name for file in changed_files.files):
                # This is not a valid patch, return 0
                print("Miner is not allowed to touch test files, returning a score of 0")
                return 0
            if isinstance(patch, str):
                return score_patch(patch, self.repo, self.row, self.docker_server._local_client, self.image_name)
            changed_files.files = [
                file for file in changed_files.files if "test" not in file.file_name
            ]
            diff = create_diff(changed_files.files)
            client = (
                self.docker_server._local_client
                if not self.use_remote or not self.docker_server.remote
                else self.docker_server._remote_client
            )
            return score_patch(diff, self.repo, self.row, client, self.image_name)
        except Exception as e:
            print("There was an error scoring the patch: ", e)
            print(traceback.format_exc())
            return 0

    def _cleanup(self):
        self.repo._cleanup()
        try:
            if self.use_remote:
                self.docker_server._local_client.images.remove(self.image_name, force=True)
                self.docker_server._remote_client.images.remove(self.image_name, force=True)
            else:
                self.docker_server._local_client.images.remove(self.image_name, force=True)
        except Exception as e:
            pass

def score_task(
    patch: Patch,
    repo_path: str,
    instance: dict,
    client: docker.DockerClient,
    image_name: str,
):
    try:
        changed_files = patch_to_changed_files(patch, repo_path)
        changed_files.files = [
            file for file in changed_files.files if "test" not in file.file_name
        ]
        print("changed_files: ", changed_files.files)
        diff = create_diff(changed_files.files)
        return score_patch(diff, instance, client, image_name)
    except Exception as e:
        print("There was an error scoring the patch-: ", e)
        print(traceback.format_exc())
        return 0
