"""IBM Spectrum LSF adaptor for SkyPilot.

This module provides ``LsfClient``, a thin control-plane client that runs
LSF batch commands (``bsub``/``bjobs``/``bkill``/``bqueues``/``bhosts``)
over SSH on an LSF login node, plus pure parser functions for their
machine-readable output.

All remote LSF commands are wrapped in a login shell (``bash -lc``) because
LSF environments are typically initialized from login-shell profile scripts
(a plain ``ssh host bjobs`` commonly fails with ``command not found``).
"""

import logging
import re
import shlex
from typing import Dict, List, NamedTuple, Optional, Tuple

from sky.utils import command_runner
from sky.utils import subprocess_utils

logger = logging.getLogger(__name__)

# Delimiter for `bjobs -o '... delimiter="|"'` output. LSF supports an
# explicit delimiter in the -o format string, which makes parsing robust
# against values containing spaces.
BJOBS_DELIMITER = '|'

# Job states, see `bjobs` documentation (JOB STATUS).
# PEND: waiting in a queue; PROV: dispatched to a power-saved host being
# woken up; WAIT: member of a chunk job waiting to run.
PENDING_STATES = frozenset({'PEND', 'PROV', 'WAIT'})
RUNNING_STATES = frozenset({'RUN'})
# Suspended, either by user/admin (PSUSP/USUSP) or by the system (SSUSP).
SUSPENDED_STATES = frozenset({'PSUSP', 'USUSP', 'SSUSP'})
# DONE: finished normally; EXIT: finished abnormally; UNKWN/ZOMBI: the
# execution host lost contact / the job is unrecoverable.
TERMINAL_STATES = frozenset({'DONE', 'EXIT', 'UNKWN', 'ZOMBI'})

# Regex to extract the job id and queue from bsub output. Site-specific esub
# scripts may PREPEND informational lines (e.g. DTU's
# `bsub info: Job has no memory requirements! ...`), so callers must scan
# every line, not just the first. When `-q` is omitted, bsub prints
# `... is submitted to default queue <...>`.
_BSUB_JOB_ID_REGEX = re.compile(
    r'Job <(?P<job_id>\d+)> is submitted to (?:default )?queue '
    r'<(?P<queue>[^>]+)>')

# `bjobs` messages that mean "no such job" rather than a real failure.
_BJOBS_NOT_FOUND_PATTERNS = (
    'is not found',
    'No unfinished job found',
    'No job found',
    'No matching job found',
)

# `bkill` messages that mean the job is already gone; treated as success.
_BKILL_ALREADY_GONE_PATTERNS = (
    'No matching job found',
    'has already finished',
    'is not found',
    'Operation is in progress',
)


class LsfQueueInfo(NamedTuple):
    """Information about an LSF queue from `bqueues -w`."""
    name: str
    priority: int
    status: str
    # Total/pending/running job slot counts in the queue.
    njobs: int
    pend: int
    run: int

    @property
    def is_open(self) -> bool:
        return self.status.startswith('Open')


class GpuHostInfo(NamedTuple):
    """One GPU on one host, from `bhosts -gpu -w`."""
    host: str
    gpu_id: int
    model: str
    # Number of jobs using / running on this GPU.
    njobs: int
    run: int

    @property
    def is_free(self) -> bool:
        return self.njobs == 0


class JobInfo(NamedTuple):
    """Basic information about an LSF job from `bjobs`."""
    job_id: str
    state: str
    # The execution host (without the leading `N*` slot multiplier),
    # or None if the job has not been dispatched yet.
    exec_host: Optional[str]
    name: str


def parse_bsub_output(output: str) -> Tuple[str, str]:
    """Parses `bsub` output into (job_id, queue).

    Scans all lines because esub scripts may prepend informational lines
    before the `Job <id> is submitted to queue <q>` line.

    Raises:
        ValueError: If no job id line is found.
    """
    for line in output.splitlines():
        match = _BSUB_JOB_ID_REGEX.search(line)
        if match is not None:
            return match.group('job_id'), match.group('queue')
    raise ValueError(f'Failed to parse job ID from bsub output: {output!r}')


def parse_exec_host(exec_host: str) -> Optional[str]:
    """Parses the first host from a bjobs EXEC_HOST value.

    LSF reports `EXEC_HOST` as `hostA` for a single-slot job, `4*hostA` for
    a multi-slot job, and `4*hostA:4*hostB` for a multi-host job. Returns
    None for an empty or `-` value (job not dispatched).
    """
    exec_host = exec_host.strip()
    if not exec_host or exec_host == '-':
        return None
    first = exec_host.split(':', 1)[0]
    if '*' in first:
        first = first.split('*', 1)[1]
    return first


def parse_bjobs_output(output: str) -> List[JobInfo]:
    """Parses `bjobs -noheader -o 'jobid stat exec_host job_name ...'`.

    The expected format string is:
        jobid stat exec_host job_name delimiter="|"
    """
    jobs = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(BJOBS_DELIMITER)
        if len(parts) < 4:
            raise RuntimeError(f'Unexpected output format from bjobs: {line!r}')
        job_id, state, exec_host, name = (p.strip() for p in parts[:4])
        jobs.append(
            JobInfo(job_id=job_id,
                    state=state,
                    exec_host=parse_exec_host(exec_host),
                    name=name))
    return jobs


def parse_bqueues_output(output: str) -> List[LsfQueueInfo]:
    """Parses `bqueues -w` output.

    Columns: QUEUE_NAME PRIO STATUS MAX JL/U JL/P JL/H NJOBS PEND RUN
    SUSP RSV [PJOBS]. STATUS is a single token (e.g. `Open:Active`).
    """
    queues = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith('QUEUE_NAME'):
            continue
        parts = line.split()
        if len(parts) < 12:
            raise RuntimeError(
                f'Unexpected output format from bqueues: {line!r}')
        try:
            queues.append(
                LsfQueueInfo(name=parts[0],
                             priority=int(parts[1]),
                             status=parts[2],
                             njobs=int(parts[7]),
                             pend=int(parts[8]),
                             run=int(parts[9])))
        except ValueError as e:
            raise RuntimeError(
                f'Failed to parse queue info from line: {line!r}. '
                f'Error: {e}') from e
    return queues


def parse_bhosts_gpu_output(output: str) -> List[GpuHostInfo]:
    """Parses `bhosts -gpu -w` output.

    Columns: HOST_NAME GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV. Hosts
    with multiple GPUs emit continuation rows with a blank HOST_NAME; the
    parser carries the previous host name forward.
    """
    gpus = []
    current_host: Optional[str] = None
    for line in output.splitlines():
        if not line.strip() or line.startswith('HOST_NAME'):
            continue
        # Continuation rows start with whitespace (blank HOST_NAME column).
        is_continuation = line[0].isspace()
        parts = line.split()
        if is_continuation:
            if current_host is None:
                raise RuntimeError(
                    f'Continuation row without a preceding host: {line!r}')
            if len(parts) < 8:
                raise RuntimeError(
                    f'Unexpected output format from bhosts -gpu: {line!r}')
            host = current_host
            gpu_id, model = parts[0], parts[1]
            njobs, run = parts[4], parts[5]
        else:
            if len(parts) < 9:
                raise RuntimeError(
                    f'Unexpected output format from bhosts -gpu: {line!r}')
            host = parts[0]
            current_host = host
            gpu_id, model = parts[1], parts[2]
            njobs, run = parts[5], parts[6]
        try:
            gpus.append(
                GpuHostInfo(host=host,
                            gpu_id=int(gpu_id),
                            model=model,
                            njobs=int(njobs),
                            run=int(run)))
        except ValueError as e:
            raise RuntimeError(f'Failed to parse GPU info from line: {line!r}. '
                               f'Error: {e}') from e
    return gpus


def _output_matches(output: str, patterns: Tuple[str, ...]) -> bool:
    return any(p in output for p in patterns)


class LsfClient:
    """Client for LSF control-plane operations on a login node.

    Unlike the Slurm equivalent, there is no local execution mode: compute
    nodes cannot be assumed to reach the LSF control plane the same way the
    login node does, and all SkyPilot control-plane operations go through
    the login node.
    """

    # LSF commands need a login shell to pick up the LSF profile
    # (e.g. /etc/profile.d/lsf.sh).
    _REMOTE_SHELL = 'bash -lc'

    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str,
        ssh_key: Optional[str],
        ssh_proxy_command: Optional[str] = None,
        ssh_proxy_jump: Optional[str] = None,
        identities_only: Optional[bool] = None,
    ):
        """Initialize LsfClient.

        Args:
            ssh_host: Hostname of the LSF login node.
            ssh_port: SSH port on the login node.
            ssh_user: SSH username.
            ssh_key: Path to SSH private key, or None for keyless SSH.
            ssh_proxy_command: Optional SSH proxy command.
            ssh_proxy_jump: Optional SSH proxy jump destination.
            identities_only: If True, only use the specified identity file
                and don't try ssh-agent keys.
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.ssh_proxy_command = ssh_proxy_command
        self.ssh_proxy_jump = ssh_proxy_jump

        self._runner = command_runner.SSHCommandRunner(
            (ssh_host, ssh_port),
            ssh_user,
            ssh_key,
            ssh_proxy_command=ssh_proxy_command,
            ssh_proxy_jump=ssh_proxy_jump,
            enable_interactive_auth=True,
            disable_identities_only=not identities_only,
        )

    @property
    def runner(self) -> command_runner.SSHCommandRunner:
        """The login node command runner (e.g. for rsync of job scripts)."""
        return self._runner

    def _run_lsf_cmd(self, cmd: str) -> Tuple[int, str, str]:
        """Runs a command in a login shell on the login node."""
        wrapped = f'{self._REMOTE_SHELL} {shlex.quote(cmd)}'
        return self._runner.run(wrapped,
                                require_outputs=True,
                                separate_stderr=True,
                                stream_logs=False)

    def run_raw(self, cmd: str) -> Tuple[int, str, str]:
        """Runs an arbitrary command in a login shell on the login node."""
        return self._run_lsf_cmd(cmd)

    def check_reachable(self) -> str:
        """Checks LSF availability by running `lsid`.

        Returns:
            The stdout of `lsid` (cluster identification).
        """
        cmd = 'lsid'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(
            rc,
            cmd,
            'Failed to reach the LSF cluster (lsid).',
            stderr=f'{stdout}\n{stderr}',
            stream_logs=False)
        return stdout

    def submit_job(self, script_path: str) -> Tuple[str, str]:
        """Submits an LSF job script with bsub.

        The script is fed to bsub on stdin (the canonical LSF submission
        mode; `#BSUB` directives in the script carry the resource request).

        Args:
            script_path: Remote path of the job script on the login node.

        Returns:
            Tuple of (job_id, queue) parsed from the bsub output.
        """
        cmd = f'bsub < {shlex.quote(script_path)}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to submit LSF job.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)
        # esub info lines may precede the submission line and may be printed
        # to either stream; scan both.
        job_id, queue = parse_bsub_output(f'{stdout}\n{stderr}')
        logger.debug(f'Successfully submitted LSF job {job_id} to queue '
                     f'{queue}: {stdout}')
        return job_id, queue

    def _bjobs(self, args: str) -> List[JobInfo]:
        """Runs bjobs with the standard machine-readable format string."""
        fmt = f'jobid stat exec_host job_name delimiter="{BJOBS_DELIMITER}"'
        cmd = f'bjobs -noheader -o {shlex.quote(fmt)} {args}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            if _output_matches(f'{stdout}\n{stderr}',
                               _BJOBS_NOT_FOUND_PATTERNS):
                return []
            subprocess_utils.handle_returncode(rc,
                                               cmd,
                                               'Failed to query LSF jobs.',
                                               stderr=f'{stdout}\n{stderr}',
                                               stream_logs=False)
        if _output_matches(stdout, _BJOBS_NOT_FOUND_PATTERNS):
            # Some LSF versions print the "not found" message on stdout with
            # a zero exit code.
            return []
        return parse_bjobs_output(stdout)

    def query_jobs_by_name(self,
                           job_name: str,
                           include_finished: bool = False) -> List[JobInfo]:
        """Queries LSF jobs by job name.

        Args:
            job_name: The LSF job name (`-J`) to filter by.
            include_finished: Also include recently finished jobs
                (DONE/EXIT within LSF's CLEAN_PERIOD).
        """
        flags = '-a ' if include_finished else ''
        return self._bjobs(f'{flags}-J {shlex.quote(job_name)}')

    def get_job(self,
                job_id: str,
                include_finished: bool = True) -> Optional[JobInfo]:
        """Returns info for a single job, or None if not found."""
        flags = '-a ' if include_finished else ''
        jobs = self._bjobs(f'{flags}{shlex.quote(job_id)}')
        if not jobs:
            return None
        return jobs[0]

    def get_job_state(self, job_id: str) -> Optional[str]:
        """Returns the state of an LSF job, or None if not found."""
        job = self.get_job(job_id)
        return job.state if job is not None else None

    def get_job_pend_reason(self, job_id: str) -> Optional[str]:
        """Returns the pending reason of a job (best-effort)."""
        cmd = (f'bjobs -noheader -o \'pend_reason\' '
               f'{shlex.quote(job_id)}')
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            return None
        reason = stdout.strip()
        if not reason or reason == '-':
            return None
        return reason

    def kill_jobs_by_name(self, job_name: str, force: bool = False) -> None:
        """Kills LSF job(s) by name.

        `bkill -J <name> 0` kills all jobs with the given name (a bare
        `bkill -J <name>` only kills the most recently submitted one).
        Jobs that are already finished or missing are treated as success.

        Args:
            job_name: Name of the job(s) to kill.
            force: If True, use `bkill -r` to force-remove the job(s) from
                LSF even if the job cannot be reached (e.g. host down).
        """
        force_flag = '-r ' if force else ''
        cmd = f'bkill {force_flag}-J {shlex.quote(job_name)} 0'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc != 0:
            output = f'{stdout}\n{stderr}'
            if _output_matches(output, _BKILL_ALREADY_GONE_PATTERNS):
                logger.debug(f'Job {job_name} already finished or not found: '
                             f'{output}')
                return
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to kill job {job_name}.',
                stderr=output,
                stream_logs=False)
        logger.debug(f'Successfully killed job {job_name}: {stdout}')

    def get_queues_info(self) -> List[LsfQueueInfo]:
        """Returns queue information from `bqueues -w`."""
        cmd = 'bqueues -w'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        subprocess_utils.handle_returncode(rc,
                                           cmd,
                                           'Failed to get LSF queues.',
                                           stderr=f'{stdout}\n{stderr}',
                                           stream_logs=False)
        return parse_bqueues_output(stdout)

    def get_gpu_hosts(self) -> List[GpuHostInfo]:
        """Returns per-GPU host information from `bhosts -gpu -w`.

        Returns an empty list if the cluster has no GPU hosts (bhosts
        errors out in that case).
        """
        cmd = 'bhosts -gpu -w'
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            logger.debug(f'bhosts -gpu failed (no GPU hosts?): {stdout}')
            return []
        return parse_bhosts_gpu_output(stdout)

    def get_env(self) -> Dict[str, str]:
        """Fetches environment variables from the login node."""
        rc, stdout, stderr = self._run_lsf_cmd('env')
        if rc != 0:
            logger.warning(f'Failed to fetch remote env: {stderr}')
            return {}
        env: Dict[str, str] = {}
        for line in stdout.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                env[key] = value
        return env

    def get_remote_home_dir(self) -> str:
        """Returns the remote user's home directory."""
        return self._runner.get_remote_home_dir()

    def check_file_exists(self, path: str) -> bool:
        """Checks if a file exists on the login node."""
        cmd = f'test -f {shlex.quote(path)}'
        rc, stdout, stderr = self._run_lsf_cmd(cmd)
        if rc not in (0, 1):
            subprocess_utils.handle_returncode(
                rc,
                cmd,
                f'Failed to check for file: {path}',
                stderr=f'{stdout}\n{stderr}')
        return rc == 0

    def read_file(self, path: str) -> Optional[str]:
        """Returns the contents of a file on the login node, or None."""
        cmd = f'cat {shlex.quote(path)}'
        rc, stdout, _ = self._run_lsf_cmd(cmd)
        if rc != 0:
            return None
        return stdout
