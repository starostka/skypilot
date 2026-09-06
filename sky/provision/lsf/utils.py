"""LSF utilities for SkyPilot."""
import math
import re
import shlex
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import lsf
from sky.provision.lsf import creds
from sky.utils import annotations
from sky.utils import common_utils
from sky.utils import gpu_names

logger = sky_logging.init_logger(__name__)

DEFAULT_LSF_PATH = creds.DEFAULT_LSF_PATH

LSF_MARKER_FILE = '.sky_lsf_cluster'

_VAR_PATTERN = re.compile(r'\$(\w+|\{[^}]*\})')
# Same shape as the Slurm backend's check: a POSIX-ish login name.
_LSF_USER_PATTERN = re.compile(r'^[a-z_][a-z0-9_.-]*$')

# Reverse-tunnel port selection: deterministic base derived from the LSF
# job id, with linear probing on collision (both on the compute node and
# on the login node). Must match the logic in the generated bsub script.
TUNNEL_PORT_BASE = 30000
TUNNEL_PORT_RANGE = 20000
TUNNEL_PORT_MAX_ATTEMPTS = 20

# Accepted `bsub -W` walltime formats: minutes, or hours:minutes.
_WALLTIME_FORMAT_RE = re.compile(r'\d+|\d+:\d{1,2}')

DEFAULT_WALLTIME = '24:00'


def expand_path_vars(path: str, env: Dict[str, str]) -> str:
    """Expand $VAR and ${VAR} in path using the given environment dict.

    Only $name and ${name} forms are expanded. Unknown variables are
    left unchanged.
    """

    def _repl(m: re.Match) -> str:
        name = m.group(1)
        if name.startswith('{') and name.endswith('}'):
            name = name[1:-1]
        return env.get(name, m.group(0))

    return _VAR_PATTERN.sub(_repl, path)


def validate_walltime(value: str) -> None:
    """Validate that a `bsub -W` walltime value is well-formed.

    LSF accepts minutes (`120`) or hours:minutes (`24:00`).

    Raises:
        ValueError: If the value does not match an LSF-accepted format.
    """
    if not _WALLTIME_FORMAT_RE.fullmatch(value):
        raise ValueError(f'Invalid LSF walltime {value!r}. '
                         'Accepted formats: minutes (e.g. 120) or '
                         'hours:minutes (e.g. 24:00).')


def tunnel_port_candidates(job_id: Union[str, int]) -> List[int]:
    """Deterministic reverse-tunnel port candidates for an LSF job id.

    The in-job script probes these ports in order until it finds one that
    is free on both the compute node and the login node; the port actually
    chosen is recorded in the job's endpoint file.
    """
    job_id_int = int(job_id)
    return [
        TUNNEL_PORT_BASE + (job_id_int + i) % TUNNEL_PORT_RANGE
        for i in range(TUNNEL_PORT_MAX_ATTEMPTS)
    ]


class LsfInstanceType:
    """Class to represent the "Instance Type" on an LSF cluster.

    Since LSF does not have a notion of instances, we generate virtual
    instance types that represent the resources requested by an LSF job,
    exactly like the Slurm backend does.

    The name format is "{n}CPU--{k}GB" with an optional
    "--{acc_type}:{acc_count}" suffix, e.g.:
        - 4CPU--16GB
        - 4CPU--16GB--V100:1
    """

    def __init__(self,
                 cpus: float,
                 memory: float,
                 accelerator_count: Optional[int] = None,
                 accelerator_type: Optional[str] = None):
        self.cpus = cpus
        self.memory = memory
        self.accelerator_count = accelerator_count
        self.accelerator_type = accelerator_type

    @property
    def name(self) -> str:
        """Returns the name of the instance."""
        assert self.cpus is not None
        assert self.memory is not None
        name = (f'{common_utils.format_float(self.cpus)}CPU--'
                f'{common_utils.format_float(self.memory)}GB')
        if self.accelerator_count is not None:
            assert self.accelerator_type is not None, self.accelerator_count
            acc_name = self.accelerator_type.replace(' ', '_')
            name += f'--{acc_name}:{self.accelerator_count}'
        return name

    @staticmethod
    def is_valid_instance_type(name: str) -> bool:
        """Returns whether the given name is a valid instance type."""
        pattern = re.compile(
            r'^(\d+(\.\d+)?CPU--\d+(\.\d+)?GB)(--[\w\d-]+:\d+)?$')
        return bool(pattern.match(name))

    @classmethod
    def _parse_instance_type(
            cls,
            name: str) -> Tuple[float, float, Optional[int], Optional[str]]:
        pattern = re.compile(
            r'^(?P<cpus>\d+(\.\d+)?)CPU--(?P<memory>\d+(\.\d+)?)GB(?:--(?P<accelerator_type>[\w\d-]+):(?P<accelerator_count>\d+))?$'  # pylint: disable=line-too-long
        )
        match = pattern.match(name)
        if match is not None:
            cpus = float(match.group('cpus'))
            memory = float(match.group('memory'))
            accelerator_count = match.group('accelerator_count')
            accelerator_type = match.group('accelerator_type')
            if accelerator_count is not None:
                accelerator_count = int(accelerator_count)
                accelerator_type = str(accelerator_type).replace(' ', '_')
            else:
                accelerator_count = None
                accelerator_type = None
            return cpus, memory, accelerator_count, accelerator_type
        else:
            raise ValueError(f'Invalid instance name: {name}')

    @classmethod
    def from_instance_type(cls, name: str) -> 'LsfInstanceType':
        """Returns an instance name object from the given name."""
        if not cls.is_valid_instance_type(name):
            raise ValueError(f'Invalid instance name: {name}')
        cpus, memory, accelerator_count, accelerator_type = \
            cls._parse_instance_type(name)
        return cls(cpus=cpus,
                   memory=memory,
                   accelerator_count=accelerator_count,
                   accelerator_type=accelerator_type)

    @classmethod
    def from_resources(cls,
                       cpus: float,
                       memory: float,
                       accelerator_count: Union[float, int] = 0,
                       accelerator_type: str = '') -> 'LsfInstanceType':
        """Returns an instance name object from the given resources."""
        name = f'{cpus}CPU--{memory}GB'
        # GPU requests in LSF must be integers.
        accelerator_count = math.ceil(accelerator_count)
        if accelerator_count > 0:
            name += f'--{accelerator_type}:{accelerator_count}'
        return cls(cpus=cpus,
                   memory=memory,
                   accelerator_count=accelerator_count,
                   accelerator_type=accelerator_type)

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'LsfInstanceType(cpus={self.cpus!r}, '
                f'memory={self.memory!r}, '
                f'accelerator_count={self.accelerator_count!r}, '
                f'accelerator_type={self.accelerator_type!r})')


def instance_id(job_id: str) -> str:
    """Generates the SkyPilot-defined instance ID for LSF.

    An LSF job id is unique within an LSF cluster. The LSF backend runs
    single-host jobs (span[hosts=1]), so the job id alone identifies the
    virtual instance.
    """
    return f'job{job_id}'


def get_lsf_cluster_from_config(provider_config: Dict[str, Any]) -> str:
    """Return the LSF cluster alias from the provider config."""
    lsf_cluster = provider_config.get('cluster')
    if lsf_cluster is None:
        raise ValueError('LSF cluster not specified in provider config.')
    return lsf_cluster


def get_queue_from_config(provider_config: Dict[str, Any]) -> str:
    """Return the queue from the provider config.

    The concept of an LSF queue maps to a cloud zone (as Slurm partitions
    do for the Slurm backend).
    """
    queue = provider_config.get('queue')
    if queue is None:
        raise ValueError('Queue not specified in provider config.')
    return queue


def get_all_lsf_cluster_names() -> List[str]:
    """Get all LSF cluster aliases available in the environment.

    Returns:
        List[str]: The list of LSF cluster aliases if available,
            an empty list otherwise.
    """
    try:
        return creds.get_provider().list_clusters()
    except FileNotFoundError:
        return []
    except Exception as e:
        raise ValueError(
            f'Failed to load LSF configuration from {DEFAULT_LSF_PATH}: '
            f'{common_utils.format_exception(e)}') from e


def get_submit_user(cluster_name: str) -> Optional[str]:
    """Returns the calling SkyPilot user's Unix account for LSF submission.

    Mirrors sky/provision/slurm/utils.py:get_submit_user. Gated on
    `lsf.cluster_configs.<cluster>.submit_as_user`, which the schema has
    always declared and nothing has ever read — so every LSF action ran as
    whatever single account the credential source named, regardless of who
    asked for it.

    The name comes from the calling user's identity rather than the process
    environment, so a multi-user API server resolves a different account per
    caller instead of one account per deployment.

    Returns None when the flag is off, meaning "use whatever user the
    credential source specifies" — the previous behaviour.
    """
    enabled = skypilot_config.get_effective_region_config(
        cloud='lsf',
        region=cluster_name,
        keys=('submit_as_user',),
        default_value=False)
    if not enabled:
        return None

    user_name = common_utils.get_current_user_name()
    # SSO identities are email-shaped; the cluster account is the local part,
    # matching how the Slurm backend derives it. Deployments that broker SSH
    # certificates should sign for this same principal — if the certificate
    # principal and the login account disagree, the certificate authenticates
    # as nobody.
    submit_user = user_name.split('@', 1)[0]
    if _LSF_USER_PATTERN.fullmatch(submit_user) is None:
        raise ValueError(
            f'Cannot derive a valid Unix user from SkyPilot user '
            f'{user_name!r}. LSF submit users must start with a lowercase '
            'letter or "_" and contain only lowercase letters, digits, "_", '
            '".", or "-".')
    return submit_user


def get_lsf_credentials(cluster: str) -> creds.LsfCredentials:
    """Resolve login-node SSH credentials for an LSF cluster alias.

    When `submit_as_user` is on, the account is taken from the CALLING user
    rather than from the credential source. This is the single chokepoint for
    that decision — anything building a client from a raw ssh dict bypasses it
    (see make_client_from_ssh_config).
    """
    credentials = creds.get_provider().get_credentials(cluster)
    submit_user = get_submit_user(cluster)
    user = submit_user or credentials.user
    if user is None:
        raise ValueError(
            f'No account for LSF cluster {cluster!r}: the credential source '
            'specifies no User and submit_as_user is not enabled. Set one or '
            'the other.')
    if user != credentials.user:
        credentials = credentials._replace(user=user)
    return credentials


def make_client(cluster: str) -> lsf.LsfClient:
    """Build an LsfClient for the given cluster alias."""
    credentials = get_lsf_credentials(cluster)
    return lsf.LsfClient(
        credentials.host,
        credentials.port,
        credentials.user,
        credentials.identity_file,
        ssh_proxy_command=credentials.proxy_command,
        ssh_proxy_jump=credentials.proxy_jump,
        identities_only=credentials.identities_only,
    )


def make_client_from_ssh_config(
        ssh_config_dict: Dict[str, Any],
        cluster: Optional[str] = None) -> lsf.LsfClient:
    """Build an LsfClient from a provider config `ssh` dict.

    `cluster` is what makes this honour submit_as_user. The provider config is
    written once at provision time and carries a FIXED user, so without the
    cluster alias every call here runs as that one account no matter who is
    asking — which is the confused-deputy shape this whole change exists to
    remove. Callers that have the alias must pass it.
    """
    user = ssh_config_dict['user']
    if cluster is not None:
        submit_user = get_submit_user(cluster)
        if submit_user is not None:
            user = submit_user
    else:
        # No alias means there is no per-cluster config to consult, so the
        # credential source's user stands — the behaviour before submit_as_user
        # existed. Logged rather than silent: on a deployment that HAS enabled
        # submit_as_user this is the one path that would still run as the
        # shared account, and it should be findable.
        logger.debug('LSF: no cluster alias supplied; submit_as_user cannot '
                     'be applied and the configured user %r is used.',
                     user)
    return lsf.LsfClient(
        ssh_config_dict['hostname'],
        int(ssh_config_dict['port']),
        user,
        ssh_config_dict.get('private_key', None),
        ssh_proxy_command=ssh_config_dict.get('proxycommand', None),
        ssh_proxy_jump=ssh_config_dict.get('proxyjump', None),
        identities_only=ssh_config_dict.get('identities_only', False),
    )


class QueueGpuInfo(NamedTuple):
    """Configured GPU shape of an LSF queue."""
    # Canonical GPU name (e.g. 'V100'), or None for a CPU queue.
    gpu_type: Optional[str]
    # Max GPUs requestable per host in this queue (0 for CPU queues).
    gpu_count: int


def get_configured_queues(cluster: str) -> Dict[str, QueueGpuInfo]:
    """Return the queue -> GPU map declared in the SkyPilot config.

    LSF selects the GPU model by *queue* (sites commonly name them after the
    accelerator, e.g. gpuv100/gpua100),
    and LSF itself has no first-class notion of a queue's GPU type; the
    mapping must therefore be declared in the config:

        lsf:
          cluster_configs:
            mycluster:
              queues:
                hpc: {}
                gpuv100: {gpus: V100, gpu_count: 2}
                gpua100: {gpus: A100, gpu_count: 2}

    Returns an empty dict when no queues are configured.
    """
    queues_config = skypilot_config.get_effective_region_config(
        cloud='lsf', region=cluster, keys=('queues',), default_value=None)
    if not queues_config:
        return {}
    queues: Dict[str, QueueGpuInfo] = {}
    for queue_name, queue_config in queues_config.items():
        queue_config = queue_config or {}
        gpu_type = queue_config.get('gpus')
        gpu_count = int(queue_config.get('gpu_count', 0))
        if gpu_type is not None and gpu_count <= 0:
            # A GPU queue without an explicit count defaults to 1 per host.
            gpu_count = 1
        queues[queue_name] = QueueGpuInfo(gpu_type=gpu_type,
                                          gpu_count=gpu_count)
    return queues


@annotations.lru_cache(scope='request')
def get_queues(cluster: str) -> List[str]:
    """Get queue names available on an LSF cluster.

    Queues declared in the SkyPilot config are authoritative (and define
    their GPU shapes); when none are declared, fall back to the open
    queues reported live by `bqueues -w`.
    """
    configured = get_configured_queues(cluster)
    if configured:
        return list(configured.keys())
    client = make_client(cluster)
    return [q.name for q in client.get_queues_info() if q.is_open]


def queues_for_accelerator(cluster: str, acc_type: str,
                           acc_count: int) -> List[str]:
    """Queues on the cluster that offer the requested accelerator."""
    matches = []
    for queue_name, info in get_configured_queues(cluster).items():
        if info.gpu_type is None:
            continue
        if info.gpu_type.lower() != acc_type.lower():
            continue
        if info.gpu_count < acc_count:
            continue
        matches.append(queue_name)
    return matches


def cpu_queues(cluster: str) -> List[str]:
    """Queues on the cluster suitable for CPU-only jobs.

    With a configured queue map these are the queues without a GPU type;
    without one, every open queue is considered (LSF cannot report a
    queue's GPU model, so nothing better is possible).
    """
    configured = get_configured_queues(cluster)
    if configured:
        return [q for q, info in configured.items() if info.gpu_type is None]
    return get_queues(cluster)


def check_instance_fits(cluster: str, instance_type: str,
                        queue: str) -> Tuple[bool, Optional[str]]:
    """Check if the given instance type fits the given cluster/queue.

    LSF exposes no per-host CPU/memory inventory in a machine-readable,
    queue-scoped way that is portable across sites, so the check is based
    on the configured queue map: accelerator type and count are validated
    against the queue's declared GPU shape; CPU and memory requests are
    left to the LSF scheduler.
    """
    inst = LsfInstanceType.from_instance_type(instance_type)
    acc_count = (inst.accelerator_count
                 if inst.accelerator_count is not None else 0)
    acc_type = inst.accelerator_type

    configured = get_configured_queues(cluster)
    queue_info = configured.get(queue)

    if acc_type is None:
        # CPU-only job: any queue works, but avoid dedicated GPU queues
        # when the queue map is configured.
        if queue_info is not None and queue_info.gpu_type is not None:
            return False, (f'Queue {queue!r} is a GPU queue '
                           f'({queue_info.gpu_type}); CPU-only jobs should '
                           'use a CPU queue.')
        return True, None

    if queue_info is None:
        return False, (
            f'Queue {queue!r} has no GPU declaration in the SkyPilot config. '
            f'Declare lsf.cluster_configs.{cluster}.queues.{queue}.gpus '
            'to run GPU jobs on it.')
    if queue_info.gpu_type is None:
        return False, f'Queue {queue!r} is a CPU queue.'
    if queue_info.gpu_type.lower() != acc_type.lower():
        return False, (f'Queue {queue!r} offers {queue_info.gpu_type}, '
                       f'not {acc_type}.')
    if queue_info.gpu_count < acc_count:
        return False, (f'Queue {queue!r} offers at most '
                       f'{queue_info.gpu_count}x {queue_info.gpu_type} '
                       f'per host; {acc_count} requested.')
    return True, None


# Vendor prefixes stripped during normalization for matching purposes.
_GPU_VENDOR_PREFIXES = ('nvidia', 'amd', 'intel', 'tesla')


def _normalize_gpu_name(name: str) -> str:
    """Normalize a GPU name for fuzzy comparison.

    Strips vendor prefixes, normalizes separators, and lowercases. Used
    only for matching, never for submission.

    Examples:
        'TeslaV100_PCIE_32GB' -> 'v100-pcie-32gb'
        'NVIDIAA100_PCIE_40GB' -> 'a100-pcie-40gb'
        'H100'                -> 'h100'
    """
    result = name.lower().replace('_', '-')
    for prefix in _GPU_VENDOR_PREFIXES:
        if result.startswith(prefix + '-'):
            result = result[len(prefix) + 1:]
            break
        # LSF GPU models frequently glue the vendor prefix onto the model
        # name without a separator (e.g. 'TeslaV100_PCIE_32GB',
        # 'NVIDIAA100_PCIE_40GB').
        if result.startswith(prefix) and len(result) > len(prefix):
            result = result[len(prefix):]
            break
    return result


def _is_segment_subsequence(segments_a: List[str],
                            segments_b: List[str]) -> bool:
    """Check if segments_a appears as an ordered subsequence of segments_b.

    Each segment must match exactly (preventing e.g. 'l4' matching 'l40').
    """
    b_iter = iter(segments_b)
    for seg in segments_a:
        for b_seg in b_iter:
            if seg == b_seg:
                break
        else:
            return False
    return True


def _is_run_together_prefix(can_norm: str, raw_norm: str) -> bool:
    """Whether a canonical name opens a run-together model name.

    Some sites report models with no separators at all ('NVIDIAH100PCIE'),
    which leaves the segment matcher above with a single segment and nothing
    to line up against. Matching the concatenated canonical name as a prefix
    handles those, and the trailing-digit guard keeps the same promise the
    segment matcher makes: 'A10' must not claim 'NVIDIAA10080GBPCIE', which
    is an A100-80GB.
    """
    can_flat = can_norm.replace('-', '')
    raw_flat = raw_norm.replace('-', '')
    if not raw_flat.startswith(can_flat):
        return False
    return not raw_flat[len(can_flat):][:1].isdigit()


def canonicalize_lsf_gpu_model(raw_name: str) -> str:
    """Convert an LSF `bhosts -gpu` MODEL string to a canonical GPU name.

    Iterates CANONICAL_GPU_NAMES (most-specific first) and returns the
    first canonical name whose normalized form matches the raw string.
    Falls back to uppercasing.

    The name matters beyond display: `lsf_catalog` groups hosts by it, so a
    model that fails to canonicalize is advertised under its raw LSF string
    and `--gpus H100` then matches nothing on the cluster.

    Examples:
        'TeslaV100_PCIE_32GB' -> 'V100-32GB'
        'NVIDIAA100_PCIE_40GB' -> 'A100'
        'NVIDIAL40S'           -> 'L40S'
        'NVIDIAH100PCIE'       -> 'H100'
        'NVIDIAA10080GBPCIE'   -> 'A100-80GB'
    """
    raw_norm = _normalize_gpu_name(raw_name)
    raw_segments = raw_norm.split('-')

    for canonical in gpu_names.CANONICAL_GPU_NAMES:
        can_norm = _normalize_gpu_name(canonical)
        if can_norm == raw_norm:
            return canonical
        can_segments = can_norm.split('-')
        if len(can_segments) < len(raw_segments):
            if _is_segment_subsequence(can_segments, raw_segments):
                return canonical
        elif _is_run_together_prefix(can_norm, raw_norm):
            return canonical

    return raw_name.upper()


def build_login_proxy_command(ssh_config_dict: Dict[str, Any]) -> str:
    """ProxyCommand that hops through the LSF login node.

    The compute-node sshd is only reachable on the login node's loopback
    (the reverse tunnel binds 127.0.0.1), so every connection to the
    virtual instance is proxied with `ssh -W` via the login node. Used
    both by the provisioner's command runners and by the generated
    `~/.sky/generated` SSH config (`ssh <cluster>`).
    """
    parts = [
        'ssh',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=30',
        # This runs during provisioning and inside background daemons, none
        # of which have a tty: without BatchMode a missing credential turns
        # into a password prompt that blocks forever instead of failing.
        '-o', 'BatchMode=yes',
        '-p', str(ssh_config_dict['port']),
    ]  # yapf: disable
    private_key = ssh_config_dict.get('private_key')
    if private_key is not None:
        # IdentitiesOnly only makes sense alongside an explicit key. Setting
        # it without one tells ssh to ignore the agent as well, leaving no
        # identity to offer at all -- which is the common case on HPC, where
        # the login node is reached with an agent-held key and ~/.lsf/config
        # carries no IdentityFile.
        parts += ['-o', 'IdentitiesOnly=yes', '-i', private_key]
    proxy_command = ssh_config_dict.get('proxycommand')
    if proxy_command is not None:
        parts += ['-o', f'ProxyCommand={proxy_command}']
    proxy_jump = ssh_config_dict.get('proxyjump')
    if proxy_jump is not None:
        parts += ['-J', proxy_jump]
    parts += [
        '-W', '%h:%p',
        f'{ssh_config_dict["user"]}@{ssh_config_dict["hostname"]}',
    ]  # yapf: disable
    return shlex.join(parts)


def get_default_walltime(cluster: str) -> str:
    """Return the default `-W` walltime for jobs on the given cluster."""
    walltime = skypilot_config.get_effective_region_config(
        cloud='lsf',
        region=cluster,
        keys=('default_walltime',),
        default_value=DEFAULT_WALLTIME)
    validate_walltime(str(walltime))
    return str(walltime)
