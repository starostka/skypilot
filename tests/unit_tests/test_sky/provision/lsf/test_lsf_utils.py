"""Unit tests for LSF provisioner utilities."""

import unittest.mock as mock

import pytest

from sky.adaptors import lsf
from sky.provision.lsf import instance as lsf_instance
from sky.provision.lsf import utils as lsf_utils
from sky.utils import status_lib


class TestLsfInstanceType:

    def test_cpu_only(self):
        inst = lsf_utils.LsfInstanceType(4, 16)
        assert inst.name == '4CPU--16GB'
        parsed = lsf_utils.LsfInstanceType.from_instance_type('4CPU--16GB')
        assert parsed.cpus == 4
        assert parsed.memory == 16
        assert parsed.accelerator_count is None

    def test_with_accelerator(self):
        inst = lsf_utils.LsfInstanceType.from_resources(8, 32, 2, 'V100')
        assert inst.name == '8CPU--32GB--V100:2'
        parsed = lsf_utils.LsfInstanceType.from_instance_type(
            '8CPU--32GB--V100:2')
        assert parsed.accelerator_type == 'V100'
        assert parsed.accelerator_count == 2

    def test_invalid(self):
        assert not lsf_utils.LsfInstanceType.is_valid_instance_type(
            'm5.4xlarge')
        with pytest.raises(ValueError):
            lsf_utils.LsfInstanceType.from_instance_type('m5.4xlarge')


class TestTunnelPorts:

    def test_deterministic_and_in_range(self):
        candidates = lsf_utils.tunnel_port_candidates('25811003')
        assert len(candidates) == lsf_utils.TUNNEL_PORT_MAX_ATTEMPTS
        assert candidates == lsf_utils.tunnel_port_candidates(25811003)
        for port in candidates:
            assert (lsf_utils.TUNNEL_PORT_BASE <= port <
                    lsf_utils.TUNNEL_PORT_BASE + lsf_utils.TUNNEL_PORT_RANGE)
        # Linear probing.
        assert candidates[1] == candidates[0] + 1


class TestWalltime:

    def test_valid(self):
        lsf_utils.validate_walltime('120')
        lsf_utils.validate_walltime('24:00')
        lsf_utils.validate_walltime('1:30')

    def test_invalid(self):
        with pytest.raises(ValueError):
            lsf_utils.validate_walltime('1-00:00:00')
        with pytest.raises(ValueError):
            lsf_utils.validate_walltime('24:00\n')


class TestCanonicalizeGpuModel:

    def test_dtu_models(self):
        # Real MODEL strings from `bhosts -gpu -w` on DTU LSF.
        assert lsf_utils.canonicalize_lsf_gpu_model(
            'TeslaV100_PCIE_32GB') == 'V100-32GB'
        assert lsf_utils.canonicalize_lsf_gpu_model('NVIDIAL40S') == 'L40S'

    def test_fallback_uppercases(self):
        assert lsf_utils.canonicalize_lsf_gpu_model(
            'SomeUnknownGpu') == 'SOMEUNKNOWNGPU'


_QUEUES_CONFIG = {
    'hpc': {},
    'gpuv100': {
        'gpus': 'V100',
        'gpu_count': 2
    },
    'gpua100': {
        'gpus': 'A100',
        'gpu_count': 2
    },
}


def _mock_region_config(cloud, region, keys, default_value=None, **kwargs):
    del kwargs
    assert cloud == 'lsf', cloud
    del region
    if keys == ('queues',):
        return _QUEUES_CONFIG
    return default_value


class TestQueueMap:

    def test_configured_queues(self):
        with mock.patch('sky.skypilot_config.get_effective_region_config',
                        side_effect=_mock_region_config):
            queues = lsf_utils.get_configured_queues('dtu')
            assert queues['hpc'] == lsf_utils.QueueGpuInfo(None, 0)
            assert queues['gpuv100'] == lsf_utils.QueueGpuInfo('V100', 2)

    def test_queues_for_accelerator(self):
        with mock.patch('sky.skypilot_config.get_effective_region_config',
                        side_effect=_mock_region_config):
            assert lsf_utils.queues_for_accelerator('dtu', 'V100',
                                                    2) == ['gpuv100']
            assert lsf_utils.queues_for_accelerator('dtu', 'v100',
                                                    1) == ['gpuv100']
            # Count exceeds the queue's per-host GPU count.
            assert lsf_utils.queues_for_accelerator('dtu', 'V100', 4) == []
            assert lsf_utils.queues_for_accelerator('dtu', 'H100', 1) == []

    def test_cpu_queues(self):
        with mock.patch('sky.skypilot_config.get_effective_region_config',
                        side_effect=_mock_region_config):
            assert lsf_utils.cpu_queues('dtu') == ['hpc']

    def test_check_instance_fits(self):
        with mock.patch('sky.skypilot_config.get_effective_region_config',
                        side_effect=_mock_region_config):
            fits, _ = lsf_utils.check_instance_fits('dtu', '4CPU--16GB', 'hpc')
            assert fits
            fits, reason = lsf_utils.check_instance_fits(
                'dtu', '4CPU--16GB', 'gpuv100')
            assert not fits
            assert 'GPU queue' in reason
            fits, _ = lsf_utils.check_instance_fits('dtu', '8CPU--32GB--V100:2',
                                                    'gpuv100')
            assert fits
            fits, reason = lsf_utils.check_instance_fits(
                'dtu', '8CPU--32GB--V100:4', 'gpuv100')
            assert not fits
            assert 'at most' in reason
            fits, reason = lsf_utils.check_instance_fits(
                'dtu', '8CPU--32GB--A100:1', 'gpuv100')
            assert not fits
            assert 'offers V100' in reason


class TestBsubScript:

    def _build(self, **kwargs):
        defaults = dict(
            cluster_name_on_cloud='sky-abcd-user',
            queue='gpuv100',
            cpus=4,
            memory_gb=16.0,
            accelerator_count=2,
            walltime='24:00',
            login_host='hpclogin1',
            base_dir='/zhome/ab/c/12345',
            tmpdir=None,
            bsub_options={},
        )
        defaults.update(kwargs)
        # pylint: disable=protected-access
        return lsf_instance._build_bsub_script(**defaults)

    def test_bsub_header(self):
        script = self._build()
        assert '#BSUB -J sky-abcd-user\n' in script
        assert '#BSUB -q gpuv100\n' in script
        assert '#BSUB -n 4\n' in script
        assert '#BSUB -R "span[hosts=1]"\n' in script
        # 16 GB over 4 slots -> 4096 MB per slot.
        assert '#BSUB -R "rusage[mem=4096MB]"\n' in script
        assert '#BSUB -gpu "num=2:mode=exclusive_process"\n' in script
        assert '#BSUB -W 24:00\n' in script
        assert ('#BSUB -o /zhome/ab/c/12345/.sky_provision/lsf-%J.out'
                in script)

    def test_cpu_only_omits_gpu_directive(self):
        script = self._build(accelerator_count=0, queue='hpc')
        assert '#BSUB -gpu' not in script

    def test_bootstrap_body(self):
        script = self._build()
        # User-owned sshd on loopback, no PAM.
        assert '-o ListenAddress=127.0.0.1' in script
        assert '-o UsePAM=no' in script
        # Reverse tunnel to the login node with forward-failure detection.
        assert '-N -R "127.0.0.1:$1:127.0.0.1:$1" "$LOGIN_HOST"' in script
        assert 'LOGIN_HOST=hpclogin1' in script
        assert '-o ExitOnForwardFailure=yes' in script
        # Deterministic port from the job id.
        assert ('CANDIDATE=$((30000 + (LSB_JOBID + i) % 20000))' in script)
        # Endpoint recorded for the provisioner.
        assert 'endpoint.json' in script
        # Cleanup on termination.
        assert 'trap cleanup EXIT' in script
        assert "trap 'exit 0' TERM" in script

    def test_custom_bsub_options(self):
        script = self._build(bsub_options={'P': 'someproject', 'x': True})
        assert '#BSUB -P someproject' in script
        assert '#BSUB -x' in script

    def test_protected_bsub_options_ignored(self):
        script = self._build(bsub_options={'q': 'otherqueue', 'J': 'other'})
        assert '#BSUB -q otherqueue' not in script
        assert '#BSUB -J other' not in script
        assert '#BSUB -q gpuv100\n' in script

    def test_newline_injection_rejected(self):
        with pytest.raises(ValueError, match='Newline'):
            self._build(bsub_options={'P': 'x\n#BSUB -q evil'})


class TestQueryInstances:

    def _query(self, jobs):
        client = mock.MagicMock()
        client.query_jobs_by_name.return_value = jobs
        with mock.patch.object(lsf_utils,
                               'make_client_from_ssh_config',
                               return_value=client):
            return lsf_instance.query_instances(
                'cluster',
                'sky-abcd-user',
                provider_config={
                    'ssh': {
                        'hostname': 'login',
                        'port': 22,
                        'user': 'u'
                    }
                },
                non_terminated_only=False,
            )

    def test_state_mapping(self):
        jobs = [
            lsf.JobInfo('1', 'PEND', None, 'sky-abcd-user'),
            lsf.JobInfo('2', 'RUN', 'n-62-11-10', 'sky-abcd-user'),
            lsf.JobInfo('3', 'SSUSP', 'n-62-11-10', 'sky-abcd-user'),
            lsf.JobInfo('4', 'DONE', 'n-62-11-10', 'sky-abcd-user'),
            lsf.JobInfo('5', 'EXIT', 'n-62-11-10', 'sky-abcd-user'),
            lsf.JobInfo('6', 'UNKWN', 'n-62-11-10', 'sky-abcd-user'),
        ]
        statuses = self._query(jobs)
        assert statuses['job1'] == (status_lib.ClusterStatus.INIT, None)
        assert statuses['job2'] == (status_lib.ClusterStatus.UP, None)
        assert statuses['job3'] == (status_lib.ClusterStatus.INIT, None)
        assert statuses['job4'] == (None, None)
        assert statuses['job5'] == (None, None)
        assert statuses['job6'] == (None, None)

    def test_non_terminated_only_filters_terminal(self):
        client = mock.MagicMock()
        client.query_jobs_by_name.return_value = [
            lsf.JobInfo('4', 'DONE', 'n', 'sky-abcd-user'),
        ]
        with mock.patch.object(lsf_utils,
                               'make_client_from_ssh_config',
                               return_value=client):
            statuses = lsf_instance.query_instances(
                'cluster',
                'sky-abcd-user',
                provider_config={
                    'ssh': {
                        'hostname': 'login',
                        'port': 22,
                        'user': 'u'
                    }
                },
                non_terminated_only=True,
            )
        assert statuses == {}
