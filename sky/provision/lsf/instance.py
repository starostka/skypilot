"""LSF instance provisioning.

An LSF "virtual instance" is a long-running single-host LSF job, mirroring
the Slurm backend's virtual instances. The fundamental difference from the
Slurm backend is runtime access: on typical LSF sites (verified on DTU's
cluster) direct SSH from the outside to compute nodes is blocked, and
`blaunch`/`lsrun` cannot be used from outside a job. The Slurm backend's
"srun into the allocation" command path therefore has no LSF equivalent.

Instead, the bsub job script itself:

1. starts a user-owned OpenSSH sshd on 127.0.0.1:<port> on the compute
   node, authenticating against the user's own ~/.ssh/authorized_keys
   (where SkyPilot's public key is enrolled at provision time via the
   shared NFS home), and
2. opens a persistent reverse tunnel `ssh -N -R <port>:127.0.0.1:<port>`
   from the compute node to the login node (node -> login SSH is allowed),
   keeping both alive for the lifetime of the job.

The chosen port is derived deterministically from the LSF job id (with
collision retry) and recorded in an endpoint file on the shared filesystem,
which the provisioner reads over login-node SSH. SkyPilot then reaches the
job's sshd by SSH-ing to 127.0.0.1:<port> *via the login node* (a
ProxyCommand hop), which means the entire upstream SSHCommandRunner
machinery works unchanged.
"""

import json
import os
import shlex
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import colorama

from sky import exceptions
from sky import sky_logging
from sky import skypilot_config
from sky.adaptors import lsf
from sky.provision import common
from sky.provision import constants
from sky.provision.lsf import utils as lsf_utils
from sky.skylet import constants as skylet_constants
from sky.utils import command_runner
from sky.utils import common_utils
from sky.utils import rich_utils
from sky.utils import status_lib
from sky.utils import subprocess_utils
from sky.utils import timeline
from sky.utils import ux_utils

logger = sky_logging.init_logger(__name__)

PROVISION_SCRIPTS_DIRECTORY_NAME = '.sky_provision'

POLL_INTERVAL_SECONDS = 2
# How long to wait for bkill'ed jobs to leave the queue before escalating.
_JOB_TERMINATION_TIMEOUT_SECONDS = 60
# How long to give the job's TERM trap to run cleanup before escalating to
# `bkill -r` (force removal).
_TERMINATION_GRACE_PERIOD_SECONDS = 30

# bsub options that SkyPilot controls and must not be overridden by users.
_BSUB_PROTECTED_OPTIONS = frozenset({
    'J',  # job name (cluster identity)
    'q',  # queue (selected by the optimizer / config)
    'n',  # slots
    'o',  # stdout log
    'e',  # stderr log
    'gpu',  # GPU request (from the resource spec)
})


def _bsub_log_path(base_dir: str, job_id: str) -> str:
    return f'{base_dir}/{PROVISION_SCRIPTS_DIRECTORY_NAME}/lsf-{job_id}.out'


def _sky_cluster_home_dir(base_dir: str, cluster_name_on_cloud: str) -> str:
    """The SkyPilot cluster's home directory on the LSF cluster.

    This path is on the shared filesystem, visible to all nodes.
    """
    return f'{base_dir}/.sky_clusters/{cluster_name_on_cloud}'


def _lsf_state_dir(base_dir: str, cluster_name_on_cloud: str) -> str:
    """Directory holding host key, tunnel key, endpoint file and logs."""
    return f'{_sky_cluster_home_dir(base_dir, cluster_name_on_cloud)}/.lsf'


def _endpoint_file_path(base_dir: str, cluster_name_on_cloud: str) -> str:
    return f'{_lsf_state_dir(base_dir, cluster_name_on_cloud)}/endpoint.json'


def _bsub_provision_script_path(base_dir: str,
                                cluster_name_on_cloud: str) -> str:
    """Path of the bsub provision script on the login node."""
    return os.path.join(base_dir, PROVISION_SCRIPTS_DIRECTORY_NAME,
                        f'{cluster_name_on_cloud}.sh')


def _skypilot_runtime_dir(tmpdir: Optional[str],
                          cluster_name_on_cloud: str) -> str:
    """Node-local runtime directory for the SkyPilot cluster.

    Kept off the shared home so SQLite state does not live on NFS.
    """
    tmp = tmpdir if tmpdir is not None else '/tmp'
    return os.path.join(tmp, f'skypilot-lsf-{cluster_name_on_cloud}')


def _build_custom_bsub_directives(bsub_options: Dict[str, Any]) -> str:
    """Build #BSUB directive lines from user-supplied bsub_options.

    Args:
        bsub_options: Dict mapping bsub option names (without the leading
            dash, e.g. 'P', 'app', 'alloc_flags') to values.

    Returns:
        A string of #BSUB directives with a leading newline, one per line,
        or an empty string. Protected options managed by SkyPilot are
        skipped with a warning.
    """
    if not bsub_options:
        return ''

    options = dict(bsub_options)
    conflicting = set(options.keys()) & _BSUB_PROTECTED_OPTIONS
    if conflicting:
        logger.warning(
            f'{colorama.Fore.YELLOW}Ignoring protected bsub options '
            f'managed by SkyPilot: {sorted(conflicting)}. Remove them '
            f'from lsf.bsub_options in ~/.sky/config.yaml.'
            f'{colorama.Style.RESET_ALL}')
        for key in conflicting:
            del options[key]

    lines = []
    for key in sorted(options):
        value = options[key]
        if value is None or value is False:
            continue
        str_value = str(value)
        # Defense in depth: prevent script injection via directive values.
        if '\n' in key or '\n' in str_value:
            raise ValueError(
                f'Newline characters are not allowed in bsub options: '
                f'{key!r}={str_value!r}')
        if key == 'W':
            lsf_utils.validate_walltime(str_value)
        if value is True:
            lines.append(f'#BSUB -{key}')
        else:
            lines.append(f'#BSUB -{key} {value}')
    if not lines:
        return ''
    return '\n' + '\n'.join(lines)


def _wait_for_job_running(
    client: 'lsf.LsfClient',
    job_id: str,
    timeout: int,
    on_pending: Optional[Callable[[str, Optional[str]], None]],
) -> None:
    """Wait for an LSF job to reach the RUN state.

    Args:
        client: The LSF client to use for queries.
        job_id: The LSF job ID.
        timeout: Maximum time to wait in seconds. If negative, wait
            indefinitely.
        on_pending: Optional callback invoked while the job is pending,
            called with (state, pend_reason).
    """
    start_time = time.time()
    last_state = None

    while timeout < 0 or time.time() - start_time < timeout:
        state = client.get_job_state(job_id)

        if state != last_state:
            logger.debug(f'Job {job_id} state: {state}')
            last_state = state

        if state is None:
            raise RuntimeError(f'Job {job_id} not found. It may have been '
                               'killed or failed.')
        if state in lsf.TERMINAL_STATES:
            raise RuntimeError(f'Job {job_id} terminated with state {state} '
                               'before it started running.')
        if state in lsf.RUNNING_STATES:
            logger.debug(f'Job {job_id} is running')
            return
        if state in lsf.PENDING_STATES and on_pending is not None:
            try:
                reason = client.get_job_pend_reason(job_id)
                on_pending(state, reason)
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to get pending status for job '
                             f'{job_id}: {e}')

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f'Job {job_id} did not start running within '
                       f'{timeout} seconds. Last state: {last_state}')


def _wait_for_job_ready(
    client: 'lsf.LsfClient',
    job_id: str,
    ready_signal: str,
    lsf_log: str,
) -> None:
    """Wait for the in-job bootstrap (sshd + reverse tunnel) to complete.

    Polls for the ready-signal file on the shared filesystem. Fails if the
    job leaves the pending/running states before the signal appears.
    """
    poll_interval_seconds = 1

    while True:
        if client.check_file_exists(ready_signal):
            return

        job_state = client.get_job_state(job_id)
        if (job_state is None or
                job_state not in (lsf.PENDING_STATES | lsf.RUNNING_STATES)):
            raise RuntimeError(f'LSF job {job_id} exited ({job_state}) '
                               'before initialization completed. See bsub '
                               f'logs for details: {lsf_log}')

        time.sleep(poll_interval_seconds)


def _read_endpoint(client: 'lsf.LsfClient', endpoint_path: str,
                   job_id: str) -> Tuple[str, int]:
    """Read the (node, tunnel port) endpoint recorded by the job script."""
    content = client.read_file(endpoint_path)
    if content is None:
        raise RuntimeError(
            f'LSF job {job_id} is running but its endpoint file '
            f'({endpoint_path}) does not exist. The in-job sshd/tunnel '
            'bootstrap may have failed; check the job logs.')
    try:
        endpoint = json.loads(content)
        return endpoint['host'], int(endpoint['port'])
    except (ValueError, KeyError) as e:
        raise RuntimeError(
            f'Failed to parse LSF endpoint file {endpoint_path}: '
            f'{content!r}') from e


def _build_bsub_script(
    cluster_name_on_cloud: str,
    queue: str,
    cpus: int,
    memory_gb: float,
    accelerator_count: int,
    walltime: str,
    login_host: str,
    base_dir: str,
    tmpdir: Optional[str],
    bsub_options: Dict[str, Any],
) -> str:
    """Build the bsub job script for an LSF virtual instance."""
    sky_cluster_home_dir = _sky_cluster_home_dir(base_dir,
                                                 cluster_name_on_cloud)
    lsf_state_dir = _lsf_state_dir(base_dir, cluster_name_on_cloud)
    skypilot_runtime_dir = _skypilot_runtime_dir(tmpdir, cluster_name_on_cloud)
    ready_signal = f'{sky_cluster_home_dir}/.sky_bsub_ready'
    lsf_marker_file = f'{sky_cluster_home_dir}/{lsf_utils.LSF_MARKER_FILE}'
    log_path = _bsub_log_path(base_dir, '%J')

    # Memory: `rusage[mem=...]` is effectively mandatory (site esub scripts
    # warn and inject a default otherwise). LSF sites commonly interpret
    # rusage[mem] per slot/core (DTU does), so distribute the total request
    # over the requested slots. The explicit MB unit avoids ambiguity from
    # per-site LSF_UNIT_FOR_LIMITS settings.
    mem_directive = ''
    if memory_gb > 0:
        mem_mb_per_core = max(1, -(-int(memory_gb * 1024) // cpus))
        mem_directive = f'#BSUB -R "rusage[mem={mem_mb_per_core}MB]"\n'

    gpu_directive = ''
    if accelerator_count > 0:
        # The GPU model is selected by the queue; the directive only
        # carries the count.
        gpu_directive = (f'#BSUB -gpu "num={accelerator_count}:'
                         f'mode=exclusive_process"\n')

    extra_bsub_directives = _build_custom_bsub_directives(bsub_options)

    runtime_dir_env_var = skylet_constants.SKY_RUNTIME_DIR_ENV_VAR_KEY

    # pylint: disable=line-too-long
    # fmt: off
    script = f"""\
#!/bin/bash
#BSUB -J {cluster_name_on_cloud}
#BSUB -q {queue}
#BSUB -n {cpus}
#BSUB -R "span[hosts=1]"
{mem_directive}{gpu_directive}#BSUB -W {walltime}
#BSUB -o {log_path}
#BSUB -e {log_path}{extra_bsub_directives}

set -u

SKY_CLUSTER_DIR={sky_cluster_home_dir}
LSF_STATE_DIR={lsf_state_dir}
RUNTIME_DIR={skypilot_runtime_dir}
LOGIN_HOST={shlex.quote(login_host)}

cleanup() {{
    saved_exit=$?
    # The Skylet is daemonized and survives the job script; kill it
    # explicitly.
    echo "Terminating Skylet, sshd and reverse tunnel..."
    if [ -f "$RUNTIME_DIR/.sky/skylet_pid" ]; then
        kill $(cat "$RUNTIME_DIR/.sky/skylet_pid") 2>/dev/null || true
    fi
    [ -n "${{SSHD_PID:-}}" ] && kill $SSHD_PID 2>/dev/null || true
    [ -n "${{TUNNEL_PID:-}}" ] && kill $TUNNEL_PID 2>/dev/null || true
    echo "Cleaning up sky directories..."
    rm -rf "$RUNTIME_DIR"
    rm -rf "$SKY_CLUSTER_DIR"
    exit $saved_exit
}}
# Run cleanup on any exit, including bootstrap failures.
trap cleanup EXIT
# On job termination (bkill sends SIGTERM before SIGKILL), exit 0 so
# cleanup treats it as a graceful shutdown.
trap 'exit 0' TERM

# Create the cluster's directories on the shared filesystem and the
# node-local runtime directory.
mkdir -p $SKY_CLUSTER_DIR/sky_logs $SKY_CLUSTER_DIR/sky_workdir $SKY_CLUSTER_DIR/.sky $LSF_STATE_DIR
mkdir -p $RUNTIME_DIR
# Marker file to indicate we're on an LSF cluster.
touch {lsf_marker_file}
# Suppress login messages.
touch $SKY_CLUSTER_DIR/.hushlogin

# Host key for the in-job (user-owned) sshd.
[ -f $LSF_STATE_DIR/host_key ] || ssh-keygen -t ed25519 -f $LSF_STATE_DIR/host_key -N '' -q
# Tunnel key: lets this job open a reverse tunnel to the login node.
# $HOME is on a shared filesystem, so enrolling the public key here is
# immediately honored by the login node's sshd.
[ -f $LSF_STATE_DIR/tunnel_key ] || ssh-keygen -t ed25519 -f $LSF_STATE_DIR/tunnel_key -N '' -q
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qF "$(cat $LSF_STATE_DIR/tunnel_key.pub)" ~/.ssh/authorized_keys || cat $LSF_STATE_DIR/tunnel_key.pub >> ~/.ssh/authorized_keys

SSHD_BIN=$(command -v sshd || echo /usr/sbin/sshd)
if [ ! -x "$SSHD_BIN" ]; then
    echo "sshd binary not found on the compute node." >&2
    exit 1
fi

start_sshd() {{
    # sshd runs fine as an unprivileged user on a loopback high port with
    # a user-owned host key; UsePAM must be off for non-root operation.
    "$SSHD_BIN" -D -e -f /dev/null \\
        -p "$1" -o ListenAddress=127.0.0.1 \\
        -h "$LSF_STATE_DIR/host_key" \\
        -o "AuthorizedKeysFile=$HOME/.ssh/authorized_keys" \\
        -o PasswordAuthentication=no -o PubkeyAuthentication=yes \\
        -o UsePAM=no -o PidFile=none \\
        -o "SetEnv={runtime_dir_env_var}=$RUNTIME_DIR" \\
        >> $LSF_STATE_DIR/sshd.log 2>&1 &
    SSHD_PID=$!
}}

start_tunnel() {{
    # Reverse-forward the sshd port to the login node's loopback. The
    # API server reaches it by proxying through the login node.
    ssh -i "$LSF_STATE_DIR/tunnel_key" \\
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
        -o IdentitiesOnly=yes \\
        -o ExitOnForwardFailure=yes \\
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
        -N -R "127.0.0.1:$1:127.0.0.1:$1" "$LOGIN_HOST" \\
        >> $LSF_STATE_DIR/tunnel.log 2>&1 &
    TUNNEL_PID=$!
}}

# Deterministic port derived from the LSF job id, with collision retry
# (the port must be free on this node AND on the login node; the latter is
# detected via ExitOnForwardFailure).
PORT=""
for i in $(seq 0 {lsf_utils.TUNNEL_PORT_MAX_ATTEMPTS - 1}); do
    CANDIDATE=$(({lsf_utils.TUNNEL_PORT_BASE} + (LSB_JOBID + i) % {lsf_utils.TUNNEL_PORT_RANGE}))
    start_sshd $CANDIDATE
    sleep 1
    if ! kill -0 $SSHD_PID 2>/dev/null; then
        continue
    fi
    start_tunnel $CANDIDATE
    sleep 3
    if kill -0 $TUNNEL_PID 2>/dev/null; then
        PORT=$CANDIDATE
        break
    fi
    kill $SSHD_PID 2>/dev/null || true
done
if [ -z "$PORT" ]; then
    echo "Failed to find a free tunnel port after {lsf_utils.TUNNEL_PORT_MAX_ATTEMPTS} attempts." >&2
    exit 1
fi

# Record the endpoint on the shared filesystem for the provisioner.
cat > $LSF_STATE_DIR/endpoint.json <<EOF
{{"host": "$(hostname)", "port": $PORT, "job_id": "$LSB_JOBID"}}
EOF
touch {ready_signal}
echo "SkyPilot LSF bootstrap ready: $(hostname) port $PORT (job $LSB_JOBID)"

# Keep sshd and the reverse tunnel alive for the lifetime of the job.
while true; do
    kill -0 $SSHD_PID 2>/dev/null || start_sshd $PORT
    kill -0 $TUNNEL_PID 2>/dev/null || start_tunnel $PORT
    sleep 15
done
"""
    # fmt: on
    # pylint: enable=line-too-long
    return script


def _enroll_public_key(client: 'lsf.LsfClient', public_key: str) -> None:
    """Enroll SkyPilot's public key in the user's authorized_keys.

    The home directory is shared between login and compute nodes, so this
    single enrollment authorizes both the login-node ProxyCommand hop and
    the in-job sshd.
    """
    public_key = public_key.strip()
    cmd = ('mkdir -p ~/.ssh && chmod 700 ~/.ssh && '
           'touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys '
           f'&& (grep -qF {shlex.quote(public_key)} ~/.ssh/authorized_keys '
           f'|| echo {shlex.quote(public_key)} >> ~/.ssh/authorized_keys)')
    rc, stdout, stderr = client.run_raw(cmd)
    subprocess_utils.handle_returncode(
        rc,
        cmd,
        'Failed to enroll the SkyPilot public key on the LSF cluster.',
        stderr=f'{stdout}\n{stderr}')


def _resolve_base_dirs(client: 'lsf.LsfClient',
                       region: str) -> Tuple[str, Optional[str]]:
    """Resolve (base_dir, tmpdir) for the cluster, expanding remote vars."""
    workdir = skypilot_config.get_effective_region_config(cloud='lsf',
                                                          region=region,
                                                          keys=('workdir',),
                                                          default_value=None)
    tmpdir = skypilot_config.get_effective_region_config(cloud='lsf',
                                                         region=region,
                                                         keys=('tmpdir',),
                                                         default_value=None)
    if workdir is not None or tmpdir is not None:
        remote_env = client.get_env()
        if workdir is not None:
            workdir = lsf_utils.expand_path_vars(workdir, remote_env)
        if tmpdir is not None:
            tmpdir = lsf_utils.expand_path_vars(tmpdir, remote_env)
        logger.debug(f'Resolved workdir: {workdir}, tmpdir: {tmpdir}')

    base_dir = workdir if workdir is not None else client.get_remote_home_dir()
    assert os.path.isabs(base_dir), (
        f'base_dir must be absolute, got: {base_dir}')
    return base_dir, tmpdir


@timeline.event
def _create_virtual_instance(
        region: str, cluster_name: str, cluster_name_on_cloud: str,
        config: common.ProvisionConfig) -> common.ProvisionRecord:
    """Creates an LSF virtual instance from the config.

    An LSF virtual instance is created by submitting a long-running job
    with bsub, to mimic a cloud VM.
    """
    provider_config = config.provider_config
    queue = lsf_utils.get_queue_from_config(provider_config)
    client = lsf_utils.make_client_from_ssh_config(provider_config['ssh'])

    num_nodes = config.count
    if num_nodes != 1:
        raise exceptions.NotSupportedError(
            'The LSF backend currently supports single-node clusters only '
            f'(requested {num_nodes} nodes). LSF jobs are submitted with '
            'span[hosts=1].')

    provision_timeout: int = provider_config['provision_timeout']
    wait_str = ('indefinitely'
                if provision_timeout < 0 else f'for {provision_timeout}s')
    logger.debug(f'Waiting {wait_str} for job to start on queue {queue}')

    last_status_msg = None

    def _on_pending(state: str, reason: Optional[str]) -> None:
        nonlocal last_status_msg
        del state  # unused
        if reason:
            msg = f'Launching (pending: {reason})'
        else:
            msg = 'Launching'
        status_msg = ux_utils.spinner_message(msg, cluster_name=cluster_name)
        if status_msg != last_status_msg:
            rich_utils.force_update_status(status_msg)
            last_status_msg = status_msg

    base_dir, tmpdir = _resolve_base_dirs(client, region)
    sky_cluster_home_dir = _sky_cluster_home_dir(base_dir,
                                                 cluster_name_on_cloud)
    ready_signal = f'{sky_cluster_home_dir}/.sky_bsub_ready'

    # Enroll the SkyPilot public key up front: the in-job sshd and the
    # login-node proxy hop both authenticate against the (shared)
    # ~/.ssh/authorized_keys.
    public_key = provider_config.get('public_key')
    if public_key:
        _enroll_public_key(client, public_key)

    # Check if a job for this cluster already exists.
    existing_jobs = [
        j for j in client.query_jobs_by_name(cluster_name_on_cloud)
        if j.state in (lsf.PENDING_STATES | lsf.RUNNING_STATES |
                       lsf.SUSPENDED_STATES)
    ]

    if existing_jobs:
        assert len(existing_jobs) == 1, (
            f'Multiple jobs found with name {cluster_name_on_cloud}: '
            f'{existing_jobs}')
        job_id = existing_jobs[0].job_id
        logger.debug(f'Job with name {cluster_name_on_cloud} already exists '
                     f'(JOBID: {job_id})')
        _wait_for_job_running(client, job_id, provision_timeout, _on_pending)
        rich_utils.force_update_status(
            ux_utils.spinner_message('Launching', cluster_name=cluster_name))
        _wait_for_job_ready(client, job_id, ready_signal,
                            _bsub_log_path(base_dir, job_id))
        return common.ProvisionRecord(
            provider_name='lsf',
            region=region,
            zone=queue,
            cluster_name=cluster_name_on_cloud,
            head_instance_id=lsf_utils.instance_id(job_id),
            resumed_instance_ids=[],
            created_instance_ids=[])

    resources = config.node_config
    accelerator_count_raw = resources.get('accelerator_count')
    try:
        accelerator_count = int(
            accelerator_count_raw) if accelerator_count_raw is not None else 0
    except (TypeError, ValueError):
        logger.warning(
            f'Invalid accelerator_count value: {accelerator_count_raw!r}. '
            'Defaulting to 0 (no accelerators).')
        accelerator_count = 0

    # The login host as resolvable *from the compute nodes*, used by the
    # in-job reverse tunnel. Defaults to the SSH hostname; override with
    # lsf.cluster_configs.<cluster>.internal_login_host when the external
    # SSH endpoint is not resolvable inside the cluster.
    internal_login_host = skypilot_config.get_effective_region_config(
        cloud='lsf',
        region=region,
        keys=('internal_login_host',),
        default_value=None)
    if internal_login_host is None:
        internal_login_host = provider_config['ssh']['hostname']

    bsub_options = resources.get('bsub_options', {}) or {}
    walltime = resources.get('walltime')
    if walltime is None:
        walltime = lsf_utils.DEFAULT_WALLTIME
    # A user-supplied -W in bsub_options takes precedence over the
    # auto-generated walltime.
    if bsub_options.get('W') not in (None, False):
        walltime = str(bsub_options.pop('W'))
    lsf_utils.validate_walltime(str(walltime))

    provision_script = _build_bsub_script(
        cluster_name_on_cloud=cluster_name_on_cloud,
        queue=queue,
        cpus=int(float(resources['cpus'])),
        memory_gb=float(resources['memory']),
        accelerator_count=accelerator_count,
        walltime=str(walltime),
        login_host=internal_login_host,
        base_dir=base_dir,
        tmpdir=tmpdir,
        bsub_options=bsub_options,
    )

    provision_script_path = _bsub_provision_script_path(base_dir,
                                                        cluster_name_on_cloud)
    provision_scripts_dir = os.path.dirname(provision_script_path)

    cmd = f'mkdir -p {shlex.quote(provision_scripts_dir)}'
    rc, stdout, stderr = client.run_raw(cmd)
    subprocess_utils.handle_returncode(
        rc,
        cmd,
        'Failed to create provision scripts directory on login node.',
        stderr=f'{stdout}\n{stderr}')

    # Rsync the provision script to the login node.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=True) as f:
        f.write(provision_script)
        f.flush()
        client.runner.rsync(f.name,
                            provision_script_path,
                            up=True,
                            stream_logs=False)

    job_id, actual_queue = client.submit_job(provision_script_path)
    logger.debug(f'Successfully submitted LSF job {job_id} to queue '
                 f'{actual_queue} for cluster {cluster_name_on_cloud}')

    lsf_log = _bsub_log_path(base_dir, job_id)
    try:
        _wait_for_job_running(client, job_id, provision_timeout, _on_pending)
        rich_utils.force_update_status(
            ux_utils.spinner_message('Launching', cluster_name=cluster_name))
        # No timeout for the in-job bootstrap: once the job is running, the
        # scheduler wait is over; sshd + tunnel setup takes seconds.
        _wait_for_job_ready(client, job_id, ready_signal, lsf_log)
    except (RuntimeError, TimeoutError, exceptions.CommandError) as e:
        log_content = client.read_file(lsf_log)
        if log_content:
            logger.error(f'=== LSF job logs ({lsf_log}) ===\n'
                         f'{log_content}'
                         f'=== End of LSF job logs ===')
        # Clean up the failed/pending job so a retry starts fresh.
        try:
            client.kill_jobs_by_name(cluster_name_on_cloud)
        except Exception as kill_e:  # pylint: disable=broad-except
            logger.debug(f'Failed to kill job {job_id} after provision '
                         f'failure: {kill_e}')
        raise e

    return common.ProvisionRecord(
        provider_name='lsf',
        region=region,
        zone=queue,
        cluster_name=cluster_name_on_cloud,
        head_instance_id=lsf_utils.instance_id(job_id),
        resumed_instance_ids=[],
        created_instance_ids=[lsf_utils.instance_id(job_id)])


def run_instances(region: str, cluster_name: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    """Run instances for the given cluster (an LSF job in this case)."""
    return _create_virtual_instance(region, cluster_name, cluster_name_on_cloud,
                                    config)


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    """See sky/provision/__init__.py"""
    del region, cluster_name_on_cloud, state
    # We already wait for the job to be running in run_instances.


@common_utils.retry
def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional[status_lib.ClusterStatus], Optional[str]]]:
    """See sky/provision/__init__.py"""
    del cluster_name, retry_if_missing  # Unused for LSF
    assert provider_config is not None, (cluster_name_on_cloud, provider_config)
    client = lsf_utils.make_client_from_ssh_config(provider_config['ssh'])

    jobs = client.query_jobs_by_name(cluster_name_on_cloud,
                                     include_finished=True)

    statuses: Dict[str, Tuple[Optional[status_lib.ClusterStatus],
                              Optional[str]]] = {}
    for job in jobs:
        if job.state in lsf.PENDING_STATES:
            sky_status: Optional[status_lib.ClusterStatus] = (
                status_lib.ClusterStatus.INIT)
        elif job.state in lsf.RUNNING_STATES:
            sky_status = status_lib.ClusterStatus.UP
        elif job.state in lsf.SUSPENDED_STATES:
            # The job still holds (or can regain) its allocation, but the
            # runtime is not running.
            sky_status = status_lib.ClusterStatus.INIT
        else:
            # DONE / EXIT / UNKWN / ZOMBI.
            sky_status = None

        if sky_status is None and non_terminated_only:
            logger.debug(f'Job {job.job_id} is terminated ({job.state}), but '
                         'query_instances is called with '
                         'non_terminated_only=True.')
            continue
        statuses[lsf_utils.instance_id(job.job_id)] = (sky_status, None)

    return statuses


def get_cluster_info(
        region: str,
        cluster_name_on_cloud: str,
        provider_config: Optional[Dict[str, Any]] = None) -> common.ClusterInfo:
    assert provider_config is not None, cluster_name_on_cloud
    client = lsf_utils.make_client_from_ssh_config(provider_config['ssh'])

    running_jobs = [
        j for j in client.query_jobs_by_name(cluster_name_on_cloud)
        if j.state in lsf.RUNNING_STATES
    ]

    if not running_jobs:
        # No running jobs found - the cluster may be pending or terminated.
        return common.ClusterInfo(
            instances={},
            head_instance_id=None,
            provider_name='lsf',
            provider_config=provider_config,
        )
    assert len(running_jobs) == 1, (
        f'Multiple running jobs found for cluster {cluster_name_on_cloud}: '
        f'{running_jobs}')
    job = running_jobs[0]

    base_dir, _ = _resolve_base_dirs(client, region)
    endpoint_path = _endpoint_file_path(base_dir, cluster_name_on_cloud)
    node, port = _read_endpoint(client, endpoint_path, job.job_id)

    inst_id = lsf_utils.instance_id(job.job_id)
    instances = {
        inst_id: [
            common.InstanceInfo(
                instance_id=inst_id,
                # The job's sshd is reached at 127.0.0.1:<port> via a
                # ProxyCommand hop through the login node (see
                # get_command_runners).
                internal_ip='127.0.0.1',
                external_ip=None,
                ssh_port=port,
                tags={
                    constants.TAG_SKYPILOT_CLUSTER_NAME: cluster_name_on_cloud,
                    'job_id': job.job_id,
                    'node': node,
                    'tunnel_port': str(port),
                },
                node_name=node,
            )
        ]
    }

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=inst_id,
        provider_name='lsf',
        provider_config=provider_config,
    )


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """LSF virtual instances cannot be stopped."""
    raise NotImplementedError()


def _wait_for_jobs_gone(client: 'lsf.LsfClient', job_name: str,
                        timeout: float) -> bool:
    """Wait until every job with this name is terminal or gone.

    Returns False if the timeout expires first. Transient query failures
    are tolerated until the deadline.
    """
    deadline = time.time() + timeout
    while True:
        try:
            jobs: Optional[List[lsf.JobInfo]] = client.query_jobs_by_name(
                job_name, include_finished=True)
        except exceptions.CommandError as e:
            logger.debug(f'Failed to query the state of job {job_name}, '
                         f'retrying: {e}')
            jobs = None
        if jobs is not None and all(
                job.state in lsf.TERMINAL_STATES for job in jobs):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_SECONDS)


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """See sky/provision/__init__.py"""
    assert provider_config is not None, cluster_name_on_cloud

    if worker_only:
        logger.warning(
            'worker_only=True is not supported for LSF, this is a no-op.')
        return

    client = lsf_utils.make_client_from_ssh_config(provider_config['ssh'])

    jobs = client.query_jobs_by_name(cluster_name_on_cloud)
    if not jobs:
        logger.debug(f'Job for cluster {cluster_name_on_cloud} not found, '
                     'it may have been terminated.')
        return

    # bkill sends SIGINT/SIGTERM/SIGKILL in sequence; the job script's TERM
    # trap runs cleanup. "Already finished" / "not found" responses are
    # treated as success by the client.
    client.kill_jobs_by_name(cluster_name_on_cloud)

    if _wait_for_jobs_gone(client, cluster_name_on_cloud,
                           _TERMINATION_GRACE_PERIOD_SECONDS):
        return
    logger.warning(
        f'Job for cluster {cluster_name_on_cloud} did not exit within '
        f'{_TERMINATION_GRACE_PERIOD_SECONDS}s of bkill. Escalating to '
        'a forced removal (bkill -r); the cleanup in the job script may '
        'be cut short, which can leave SkyPilot runtime directories '
        'behind on the node.')
    client.kill_jobs_by_name(cluster_name_on_cloud, force=True)
    if not _wait_for_jobs_gone(client, cluster_name_on_cloud,
                               _JOB_TERMINATION_TIMEOUT_SECONDS):
        raise RuntimeError(
            f'LSF job for cluster {cluster_name_on_cloud} is still '
            f'running {_JOB_TERMINATION_TIMEOUT_SECONDS}s after bkill -r. '
            'The allocation may be leaked; check bjobs and kill the '
            'job manually if needed.')


def open_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """See sky/provision/__init__.py"""
    del cluster_name_on_cloud, ports, provider_config


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    """See sky/provision/__init__.py"""
    del cluster_name_on_cloud, ports, provider_config


def _build_login_proxy_command(ssh_config_dict: Dict[str, Any]) -> str:
    """ProxyCommand that hops through the LSF login node.

    The compute-node sshd is only reachable on the login node's loopback
    (the reverse tunnel binds 127.0.0.1), so every connection to the
    virtual instance is proxied with `ssh -W` via the login node.
    """
    parts = [
        'ssh',
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'IdentitiesOnly=yes',
        '-o', 'ExitOnForwardFailure=yes',
        '-o', 'ServerAliveInterval=30',
        '-p', str(ssh_config_dict['port']),
    ]  # yapf: disable
    private_key = ssh_config_dict.get('private_key')
    if private_key is not None:
        parts += ['-i', private_key]
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


def get_command_runners(
    cluster_info: common.ClusterInfo,
    **credentials: Dict[str, Any],
) -> List[command_runner.SSHCommandRunner]:
    """Get command runners for the given cluster.

    Returns plain SSHCommandRunners targeting the in-job sshd
    (127.0.0.1:<tunnel port>) through a login-node ProxyCommand, so all
    upstream provisioning/execution machinery works unchanged.
    """
    assert cluster_info.provider_config is not None, cluster_info

    if cluster_info.head_instance_id is None:
        # No running job found.
        return []

    provider_config = cluster_info.provider_config
    ssh_config_dict = provider_config['ssh']
    proxy_command = _build_login_proxy_command(ssh_config_dict)

    # The SkyPilot key: enrolled in the user's authorized_keys at provision
    # time, accepted by the in-job sshd (and by the login node, though the
    # proxy hop uses the configured LSF identity).
    ssh_private_key = credentials.get('ssh_private_key')

    instances = [
        instance_infos[0] for instance_infos in cluster_info.instances.values()
    ]

    runners = [
        command_runner.SSHCommandRunner(
            ('127.0.0.1', instance_info.ssh_port),
            ssh_config_dict['user'],
            ssh_private_key,
            ssh_proxy_command=proxy_command,
            # All LSF virtual instances proxy through the same login node;
            # the %C hash in ControlPath still separates control sockets
            # per (host, port, user).
            ssh_control_name=command_runner.DEFAULT_SSH_CONTROL_NAME,
            enable_interactive_auth=True,
            # Allow ssh-agent and default key fallback.
            disable_identities_only=True,
        ) for instance_info in instances
    ]

    return runners
