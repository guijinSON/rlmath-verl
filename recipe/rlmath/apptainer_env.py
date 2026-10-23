"""Harbor sandbox environment for JUPITER (JSC) GPU nodes: Harbor's own Singularity/Apptainer backend.

On JUPITER there is no Docker; tasks run as Apptainer instances on the GPU node's Grace CPUs, from one
prebuilt sandbox image (`[environment] docker_image = "<...>/rlmath-sandbox.sif"`, built from the shared
rlmath Dockerfile with a baked /opt/harbor-server venv so bootstrap.sh never needs the network). The
task copies have no environment/Dockerfile, so Harbor uploads environment/ (instance.json) into
`workdir` after start, exactly what the Dockerfile's COPY did.

Only change vs. the stock environment: declare `disable_internet`. JUPITER compute nodes have no
outbound internet at all (checked 2026-09-29: HTTPS to HF/PyPI/GitHub/Google fails, no proxy), so the
tasks' `network_mode = "no-network"` holds for every sandbox there; the stock class cannot know that and
rejects such tasks. (Sandboxes still share the node's local network namespace.)
"""
from harbor.environments.singularity.singularity import SingularityEnvironment


class RLMathApptainerEnvironment(SingularityEnvironment):
    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"disable_internet": True})

    async def _start_server(self) -> None:
        # The stock command runs `singularity exec --fakeroot ... <image>`. Without user namespaces
        # (JUPITER: max_user_namespaces=0) Apptainer only allows --fakeroot on a *sandbox directory*
        # image, not a SIF ("--fakeroot used without sandbox image or user namespaces"). Use the
        # unpacked twin `<name>.sandbox/` next to the SIF when it exists
        # (`apptainer build --sandbox <name>.sandbox <name>.sif`); task.toml keeps pointing at the SIF.
        if self._sif_path is not None:
            sandbox_dir = self._sif_path.with_suffix(".sandbox")
            if sandbox_dir.is_dir():
                self._sif_path = sandbox_dir
        await super()._start_server()

    def _resolve_workdir(self) -> str:
        # Stock code reads WORKDIR from environment/Dockerfile and otherwise falls back to /app; our task
        # copies have no Dockerfile (the image is prebuilt), so honour [environment] workdir instead.
        return self.task_env_config.workdir or super()._resolve_workdir()
