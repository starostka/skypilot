"""Unit tests for the LSF adaptor parsers and client.

The fixture strings below are captured verbatim from a live IBM Spectrum
LSF 10.1.0.15 cluster (DTU HPC), including the site-specific esub
informational lines that precede bsub's submission message and the
continuation rows in `bhosts -gpu -w` output.
"""

import unittest.mock as mock

import pytest

from sky.adaptors import lsf

# bsub output with esub-prepended info lines (DTU esub warns and injects a
# default memory request when rusage[mem=...] is omitted).
BSUB_OUTPUT_WITH_ESUB_PREFIX = """\
bsub info: Job has no memory requirements!
bsub info: Setting default: rusage[mem=1024MB]
bsub info: Please specify -R "rusage[mem=XXMB]"
Job <25811003> is submitted to queue <hpc>.
"""

BSUB_OUTPUT_PLAIN = 'Job <123> is submitted to queue <gpuv100>.\n'

BSUB_OUTPUT_DEFAULT_QUEUE = 'Job <42> is submitted to default queue <hpc>.\n'

# `bqueues -w` (subset of the DTU queue list).
BQUEUES_W_OUTPUT = """\
QUEUE_NAME      PRIO STATUS          MAX JL/U JL/P JL/H NJOBS  PEND   RUN  SUSP  RSV PJOBS
hpcint           30  Open:Active       -    -    -    -     0     0     0     0    0     0
hpc              30  Open:Active       -    -    -    -  7175  4928  2224     0   23   988
hpcintro         30  Open:Active       -   24    -    -     0     0     0     0    0     0
gpuv100          30  Open:Active       -    -    -    -   292    76   172     0   44    19
gpua100          30  Open:Active       -    -    -    -    72     8    32     0   32     1
gpul40s          30  Open:Active       -    -    -    -   100    72    28     0    0    17
computebigbigmem  30  Open:Active       -    -    -    -     0     0     0     0    0     0
"""

# `bhosts -gpu -w` with continuation rows (blank HOST_NAME carries the
# previous host).
BHOSTS_GPU_W_OUTPUT = """\
HOST_NAME            GPU_ID                MODEL     MUSED      MRSV  NJOBS    RUN   SUSP    RSV
n-62-11-10                0  TeslaV100_PCIE_32GB      273M        0M      0      0      0      0
                          1  TeslaV100_PCIE_32GB      273M        0M      0      0      0      0
n-62-11-13                0  TeslaV100_PCIE_32GB     14.5G        0M      1      1      0      0
                          1  TeslaV100_PCIE_32GB     24.5G        0M      1      1      0      0
n-62-11-14                0  TeslaV100_PCIE_32GB     24.5G        0M      1      1      0      0
                          1  TeslaV100_PCIE_32GB     14.5G        0M      1      1      0      0
"""

# `bjobs -noheader -o 'jobid stat exec_host job_name delimiter="|"'`.
BJOBS_OUTPUT = """\
25811003|RUN|4*n-62-11-10|sky-cluster-a
25811004|PEND|-|sky-cluster-b
"""


class TestParseBsubOutput:

    def test_parse_with_esub_prefix_lines(self):
        job_id, queue = lsf.parse_bsub_output(BSUB_OUTPUT_WITH_ESUB_PREFIX)
        assert job_id == '25811003'
        assert queue == 'hpc'

    def test_parse_plain(self):
        job_id, queue = lsf.parse_bsub_output(BSUB_OUTPUT_PLAIN)
        assert job_id == '123'
        assert queue == 'gpuv100'

    def test_parse_default_queue(self):
        job_id, queue = lsf.parse_bsub_output(BSUB_OUTPUT_DEFAULT_QUEUE)
        assert job_id == '42'
        assert queue == 'hpc'

    def test_parse_failure_raises(self):
        with pytest.raises(ValueError, match='Failed to parse job ID'):
            lsf.parse_bsub_output('Request aborted by esub. Job not '
                                  'submitted.')


class TestParseExecHost:

    def test_single_slot(self):
        assert lsf.parse_exec_host('n-62-11-10') == 'n-62-11-10'

    def test_multi_slot(self):
        assert lsf.parse_exec_host('4*n-62-11-10') == 'n-62-11-10'

    def test_multi_host_returns_first(self):
        assert lsf.parse_exec_host('4*hostA:4*hostB') == 'hostA'

    def test_not_dispatched(self):
        assert lsf.parse_exec_host('-') is None
        assert lsf.parse_exec_host('') is None


class TestParseBjobsOutput:

    def test_parse_running_and_pending(self):
        jobs = lsf.parse_bjobs_output(BJOBS_OUTPUT)
        assert jobs == [
            lsf.JobInfo(job_id='25811003',
                        state='RUN',
                        exec_host='n-62-11-10',
                        name='sky-cluster-a'),
            lsf.JobInfo(job_id='25811004',
                        state='PEND',
                        exec_host=None,
                        name='sky-cluster-b'),
        ]

    def test_empty_output(self):
        assert lsf.parse_bjobs_output('') == []

    def test_malformed_line_raises(self):
        with pytest.raises(RuntimeError, match='Unexpected output format'):
            lsf.parse_bjobs_output('123|RUN')


class TestParseBqueuesOutput:

    def test_parse_queues(self):
        queues = lsf.parse_bqueues_output(BQUEUES_W_OUTPUT)
        names = [q.name for q in queues]
        assert names == [
            'hpcint', 'hpc', 'hpcintro', 'gpuv100', 'gpua100', 'gpul40s',
            'computebigbigmem'
        ]

        hpc = queues[1]
        assert hpc.priority == 30
        assert hpc.status == 'Open:Active'
        assert hpc.is_open
        assert hpc.njobs == 7175
        assert hpc.pend == 4928
        assert hpc.run == 2224

    def test_header_skipped(self):
        queues = lsf.parse_bqueues_output(
            'QUEUE_NAME PRIO STATUS MAX JL/U JL/P JL/H NJOBS PEND RUN SUSP '
            'RSV PJOBS\n')
        assert queues == []


class TestParseBhostsGpuOutput:

    def test_parse_with_continuation_rows(self):
        gpus = lsf.parse_bhosts_gpu_output(BHOSTS_GPU_W_OUTPUT)
        assert len(gpus) == 6
        # Continuation rows carry the previous host name.
        assert [g.host for g in gpus] == [
            'n-62-11-10', 'n-62-11-10', 'n-62-11-13', 'n-62-11-13',
            'n-62-11-14', 'n-62-11-14'
        ]
        assert [g.gpu_id for g in gpus] == [0, 1, 0, 1, 0, 1]
        assert all(g.model == 'TeslaV100_PCIE_32GB' for g in gpus)
        # First host: both GPUs are free; the others are in use.
        assert [g.is_free for g in gpus
               ] == [True, True, False, False, False, False]

    def test_continuation_without_host_raises(self):
        bad = ('HOST_NAME GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV\n'
               '     0  TeslaV100_PCIE_32GB 273M 0M 0 0 0 0\n')
        with pytest.raises(RuntimeError, match='Continuation row'):
            lsf.parse_bhosts_gpu_output(bad)

    def test_empty_output(self):
        assert lsf.parse_bhosts_gpu_output('') == []


def _make_client() -> lsf.LsfClient:
    return lsf.LsfClient(
        ssh_host='login1.example.com',
        ssh_port=22,
        ssh_user='user1',
        ssh_key=None,
    )


class TestLsfClient:

    def test_commands_wrapped_in_login_shell(self):
        """LSF needs a login shell over SSH to find its commands."""
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (0, BQUEUES_W_OUTPUT, '')
            client.get_queues_info()
            cmd = mock_run.call_args[0][0]
            assert cmd.startswith('bash -lc ')
            assert 'bqueues -w' in cmd

    def test_submit_job_parses_esub_prefixed_output(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (0, BSUB_OUTPUT_WITH_ESUB_PREFIX, '')
            job_id, queue = client.submit_job('/home/user1/script.sh')
            assert job_id == '25811003'
            assert queue == 'hpc'
            cmd = mock_run.call_args[0][0]
            assert 'bsub < /home/user1/script.sh' in cmd

    def test_query_jobs_by_name_not_found_is_empty(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (255, '', 'Job <sky-foo> is not found\n')
            assert client.query_jobs_by_name('sky-foo') == []

    def test_get_job_state(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (
                0, '25811003|RUN|4*n-62-11-10|sky-cluster-a\n', '')
            assert client.get_job_state('25811003') == 'RUN'

    def test_kill_jobs_by_name_tolerates_finished_jobs(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (255, '',
                                     'Job <123>: Job has already finished\n')
            # Should not raise.
            client.kill_jobs_by_name('sky-cluster-a')

    def test_kill_jobs_by_name_kills_all_matching(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (0, 'Job <123> is being terminated\n', '')
            client.kill_jobs_by_name('sky-cluster-a')
            cmd = mock_run.call_args[0][0]
            # `0` kills every job with the name, not just the latest.
            assert 'bkill -J sky-cluster-a 0' in cmd

    def test_get_gpu_hosts_tolerates_failure(self):
        client = _make_client()
        with mock.patch.object(client._runner, 'run') as mock_run:
            mock_run.return_value = (255, 'No hosts found\n', '')
            assert client.get_gpu_hosts() == []


class TestStateSets:

    def test_state_partition(self):
        all_states = (lsf.PENDING_STATES | lsf.RUNNING_STATES |
                      lsf.SUSPENDED_STATES | lsf.TERMINAL_STATES)
        # No state belongs to two sets.
        total = (len(lsf.PENDING_STATES) + len(lsf.RUNNING_STATES) +
                 len(lsf.SUSPENDED_STATES) + len(lsf.TERMINAL_STATES))
        assert len(all_states) == total
