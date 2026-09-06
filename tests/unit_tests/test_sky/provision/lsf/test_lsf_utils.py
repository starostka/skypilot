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
        # Every MODEL string `bhosts -gpu -w` reports on DTU LSF.
        for raw, expected in [
            ('TeslaV100_PCIE_16GB', 'V100'),
            ('TeslaV100_PCIE_32GB', 'V100-32GB'),
            ('TeslaV100_SXM2_32GB', 'V100-32GB'),
            ('NVIDIAA100_PCIE_40GB', 'A100'),
            ('NVIDIAA10', 'A10'),
            ('NVIDIAA40', 'A40'),
            ('NVIDIAL40S', 'L40S'),
        ]:
            assert lsf_utils.canonicalize_lsf_gpu_model(raw) == expected

    def test_models_written_without_separators(self):
        # Some DTU hosts report the model as one run-together word. These used
        # to fall through to the uppercase fallback, which matters beyond
        # display: lsf_catalog groups hosts by this name, so the H100 lane
        # advertised itself as 'NVIDIAH100PCIE' and `--gpus H100` matched
        # nothing.
        assert lsf_utils.canonicalize_lsf_gpu_model('NVIDIAH100PCIe') == 'H100'
        assert lsf_utils.canonicalize_lsf_gpu_model(
            'NVIDIAA10080GBPCIe') == 'A100-80GB'

    def test_a10_does_not_claim_an_a100(self):
        # The trailing-digit guard: without it 'A10' is a prefix of
        # 'a10080gbpcie' and wins by list order.
        assert lsf_utils.canonicalize_lsf_gpu_model(
            'NVIDIAA10080GBPCIe') != 'A10'
        assert lsf_utils.canonicalize_lsf_gpu_model('NVIDIAA100PCIe') == 'A100'

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
        # The FULL request per slot, not divided across them: MEMLIMIT
        # tracks the per-slot figure, so dividing capped the job at 1/cpus of
        # what it asked for and LSF killed it with TERM_MEMLIMIT (5548644e).
        assert '#BSUB -R "rusage[mem=16384MB]"\n' in script
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


def _build_script(**kwargs):
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


class TestBsubScriptSshServerSelection:
    """The in-job SSH server must prove auth end-to-end, not just run.

    Live-verified on a DTU EL9 node: the RHEL OpenSSH build forces PAM
    ('UsePAM no' is not supported in this build), accepts the publickey
    and then rejects the session in the PAM account stage, so a
    process-liveness check would accept a server that rejects every
    login.
    """

    def test_auth_self_test_before_tunnel(self):
        script = _build_script()
        assert 'auth_self_test() {' in script
        # End-to-end probe using the already-enrolled tunnel key.
        assert ('-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5'
                in script)
        assert '127.0.0.1 true' in script
        # The self-test gates the tunnel: it must appear before
        # start_tunnel in the port loop.
        loop = script.split('for i in $(seq', 1)[1]
        assert loop.index('auth_self_test $CANDIDATE') < loop.index(
            'start_tunnel $CANDIDATE')

    def test_auth_failure_is_binary_level_not_port_level(self):
        script = _build_script()
        # On auth failure the script falls back to dropbear on the SAME
        # port; only bind/forward failures advance the port scan.
        assert 'fall back to dropbear on the SAME port' in script
        assert 'Bind failure: port taken on this node' in script
        assert 'Port taken on the login node' in script

    def test_no_fallback_configured(self):
        script = _build_script(remote_ssh_server=None)
        assert "REMOTE_SSH_SERVER=''" in script

    def test_fallback_path_embedded(self):
        script = _build_script(
            remote_ssh_server='/zhome/ab/c/12345/bin/dropbearmulti')
        assert ('REMOTE_SSH_SERVER=/zhome/ab/c/12345/bin/dropbearmulti'
                in script)

    def test_fallback_supports_file_and_directory_shapes(self):
        script = _build_script(remote_ssh_server='/some/path')
        # Directory shape: separate dropbear + dropbearkey binaries.
        assert 'if [ -d "$REMOTE_SSH_SERVER" ]; then' in script
        assert 'DROPBEAR_CMD="$REMOTE_SSH_SERVER/dropbear"' in script
        assert 'DROPBEARKEY_CMD="$REMOTE_SSH_SERVER/dropbearkey"' in script
        # File shape: dropbearmulti multi-call binary.
        assert 'DROPBEAR_CMD="$REMOTE_SSH_SERVER dropbear"' in script
        assert 'DROPBEARKEY_CMD="$REMOTE_SSH_SERVER dropbearkey"' in script

    def test_dropbear_invocation(self):
        script = _build_script(remote_ssh_server='/some/path')
        # -F foreground, -E stderr, -s no password auth, loopback bind,
        # dropbear-format host key (separate from the OpenSSH host_key).
        assert '$DROPBEAR_CMD -F -E -s -p "127.0.0.1:$1"' in script
        assert '-r "$LSF_STATE_DIR/dropbear_host_key"' in script
        assert '$DROPBEARKEY_CMD -t ed25519' in script

    def test_runtime_dir_published_via_command_wrapper(self):
        """Neither server can inject env vars (PAM-forced sshd rejects
        UsePAM=no; dropbear has no SetEnv) and EL9 bash does not source
        ~/.bashrc for non-interactive SSH commands (live-verified empty
        on DTU), so the authorized_keys command= wrapper -- honored
        identically by OpenSSH and dropbear -- resolves the runtime dir
        from the server port."""
        script = _build_script()
        assert '-o SetEnv' not in script
        assert '.bashrc' not in script.replace('does not source ~/.bashrc', '')
        assert 'echo "$RUNTIME_DIR" > ~/.sky/lsf_ports/$PORT' in script
        # Wrapper body: port -> runtime dir mapping, then exec
        # pass-through (or a login shell for interactive sessions).
        assert '_sky_lsf_port=${SSH_CONNECTION##* }' in script
        assert ('SKY_RUNTIME_DIR='
                '"$(cat "$HOME/.sky/lsf_ports/$_sky_lsf_port")"') in script
        assert 'export SKY_RUNTIME_DIR' in script
        assert 'exec /bin/sh -c "$SSH_ORIGINAL_COMMAND"' in script
        assert 'exec "${SHELL:-/bin/sh}" -l' in script
        # Cleanup removes the port map entry.
        assert 'rm -f ~/.sky/lsf_ports/$PORT' in script

    def test_tunnel_key_line_forces_wrapper(self):
        script = _build_script()
        enroll_line = ('echo "command=\\"$HOME/.sky/lsf/env_wrapper.sh\\" '
                       '$(cat $LSF_STATE_DIR/tunnel_key.pub)"')
        assert enroll_line in script
        # Idempotence keys on the key material, not the whole line.
        assert ('grep -qF "$(cat $LSF_STATE_DIR/tunnel_key.pub)" '
                '~/.ssh/authorized_keys') in script
        # The wrapper must be written before the key line that forces it.
        assert (script.index("cat > ~/.sky/lsf/env_wrapper.sh") <
                script.index(enroll_line))
        # The auth self-test uses the tunnel key, so it exercises the
        # wrapper's exec pass-through under the active server.
        assert script.index(enroll_line) < script.index('auth_self_test()')


class TestEnrollPublicKey:
    """The SkyPilot runner key must force the env wrapper via command=."""

    def _enroll(self, public_key='ssh-ed25519 AAAATESTKEY sky'):
        client = mock.MagicMock()
        client.run_raw.return_value = (0, '', '')
        # pylint: disable=protected-access
        lsf_instance._enroll_public_key(client, public_key + '\n')
        return client.run_raw.call_args.args[0]

    def test_wrapper_installed_with_key(self):
        cmd = self._enroll()
        # The wrapper is written (and made executable) in the same remote
        # command as the key line, so the two never exist without each
        # other.
        assert 'printf %s ' in cmd
        assert '> ~/.sky/lsf/env_wrapper.sh' in cmd
        assert 'chmod 755 ~/.sky/lsf/env_wrapper.sh' in cmd
        # Wrapper body ships verbatim: port map lookup + exec
        # pass-through.
        assert '_sky_lsf_port=${SSH_CONNECTION##* }' in cmd
        assert 'exec /bin/sh -c "$SSH_ORIGINAL_COMMAND"' in cmd

    def test_key_line_forces_wrapper_with_literal_home(self):
        cmd = self._enroll()
        # $HOME expands remotely at enroll time; authorized_keys command=
        # takes no variables.
        assert ('echo "command=\\"$HOME/.sky/lsf/env_wrapper.sh\\" "'
                "'ssh-ed25519 AAAATESTKEY sky' >> ~/.ssh/authorized_keys"
                in cmd)

    def test_idempotence_keys_on_key_material(self):
        cmd = self._enroll()
        # grep on the bare key, not the command=-prefixed line, so
        # re-enrollment never duplicates.
        assert "grep -qF 'ssh-ed25519 AAAATESTKEY sky' " in cmd

    def test_enroll_failure_raises(self):
        client = mock.MagicMock()
        client.run_raw.return_value = (1, '', 'permission denied')
        with pytest.raises(Exception):
            # pylint: disable=protected-access
            lsf_instance._enroll_public_key(client, 'ssh-ed25519 AAAA x')


class TestStageRemoteSshServer:

    def _client(self, remote_sizes):
        """Mocked LsfClient whose run_raw answers stat/mkdir/chmod.

        remote_sizes maps remote path -> size (or None for missing).
        """
        import shlex as _shlex

        client = mock.MagicMock()

        def _run_raw(cmd):
            if cmd.startswith('stat -c %s '):
                path = _shlex.split(cmd)[-1]
                size = remote_sizes.get(path)
                if size is None:
                    return (1, '', 'No such file or directory')
                return (0, f'{size}\n', '')
            return (0, '', '')

        client.run_raw.side_effect = _run_raw
        return client

    def test_file_shape_skips_when_size_matches(self, tmp_path):
        local = tmp_path / 'dropbearmulti'
        local.write_bytes(b'x' * 100)
        client = self._client({'/remote/bin/dropbearmulti': 100})
        # pylint: disable=protected-access
        lsf_instance._stage_remote_ssh_server(client, str(local),
                                              '/remote/bin/dropbearmulti')
        client.runner.rsync.assert_not_called()

    def test_file_shape_stages_and_marks_executable(self, tmp_path):
        local = tmp_path / 'dropbearmulti'
        local.write_bytes(b'x' * 100)
        client = self._client({'/remote/bin/dropbearmulti': None})
        # pylint: disable=protected-access
        lsf_instance._stage_remote_ssh_server(client, str(local),
                                              '/remote/bin/dropbearmulti')
        client.runner.rsync.assert_called_once_with(str(local),
                                                    '/remote/bin/dropbearmulti',
                                                    up=True,
                                                    stream_logs=False)
        run_raw_cmds = [c.args[0] for c in client.run_raw.call_args_list]
        assert 'mkdir -p /remote/bin' in run_raw_cmds
        assert 'chmod +x /remote/bin/dropbearmulti' in run_raw_cmds

    def test_directory_shape_stages_only_missing_binaries(self, tmp_path):
        (tmp_path / 'dropbear').write_bytes(b'x' * 100)
        (tmp_path / 'dropbearkey').write_bytes(b'y' * 50)
        client = self._client({
            '/remote/dropbear-bin/dropbear': 100,  # already staged
            '/remote/dropbear-bin/dropbearkey': None,
        })
        # pylint: disable=protected-access
        lsf_instance._stage_remote_ssh_server(client, str(tmp_path),
                                              '/remote/dropbear-bin')
        client.runner.rsync.assert_called_once_with(
            str(tmp_path / 'dropbearkey'),
            '/remote/dropbear-bin/dropbearkey',
            up=True,
            stream_logs=False)

    def test_directory_shape_missing_binary_raises(self, tmp_path):
        (tmp_path / 'dropbear').write_bytes(b'x')
        client = self._client({})
        with pytest.raises(ValueError, match='dropbearkey'):
            # pylint: disable=protected-access
            lsf_instance._stage_remote_ssh_server(client, str(tmp_path),
                                                  '/remote/dropbear-bin')

    def test_missing_local_path_raises(self, tmp_path):
        client = self._client({})
        with pytest.raises(ValueError, match='does not exist'):
            # pylint: disable=protected-access
            lsf_instance._stage_remote_ssh_server(client,
                                                  str(tmp_path / 'nope'),
                                                  '/remote/x')


class TestSelfTermination:
    """Autodown: the skylet tears the cluster down from inside the job.

    The SSH path cannot run there — the caller's certificate lives on the API
    server — so terminate_instances() must recognise the allocation it is
    standing in and use `bkill` locally.
    """

    @staticmethod
    def _completed(returncode: int = 0, stderr: str = ''):
        return mock.Mock(returncode=returncode, stdout='', stderr=stderr)

    def test_bkill_issued_inside_own_allocation(self, monkeypatch):
        monkeypatch.setenv('LSB_JOBID', '4242')
        monkeypatch.setenv('LSB_JOBNAME', 'sky-abcd-user')
        monkeypatch.delenv('LSF_BINDIR', raising=False)
        run = mock.Mock(return_value=self._completed())
        monkeypatch.setattr(lsf_instance.subprocess, 'run', run)

        # pylint: disable=protected-access
        assert lsf_instance._terminate_from_inside_allocation(
            'sky-abcd-user') is True
        assert run.call_args[0][0] == ['bkill', '4242']

    def test_falls_back_to_lsf_bindir(self, monkeypatch):
        monkeypatch.setenv('LSB_JOBID', '7')
        monkeypatch.delenv('LSB_JOBNAME', raising=False)
        monkeypatch.setenv('LSF_BINDIR', '/opt/lsf/bin')
        run = mock.Mock(side_effect=[
            OSError('bkill: not found'),
            self._completed(),
        ])
        monkeypatch.setattr(lsf_instance.subprocess, 'run', run)

        # pylint: disable=protected-access
        assert lsf_instance._terminate_from_inside_allocation('sky-x') is True
        assert run.call_args[0][0] == ['/opt/lsf/bin/bkill', '7']

    def test_raises_when_no_bkill_works(self, monkeypatch):
        monkeypatch.setenv('LSB_JOBID', '7')
        monkeypatch.delenv('LSB_JOBNAME', raising=False)
        monkeypatch.delenv('LSF_BINDIR', raising=False)
        monkeypatch.setattr(
            lsf_instance.subprocess, 'run',
            mock.Mock(return_value=self._completed(1, 'Job not found')))

        with pytest.raises(RuntimeError, match='leaked'):
            # pylint: disable=protected-access
            lsf_instance._terminate_from_inside_allocation('sky-x')

    def test_declines_another_jobs_allocation(self, monkeypatch):
        monkeypatch.setenv('LSB_JOBID', '9')
        monkeypatch.setenv('LSB_JOBNAME', 'someone-elses-job')
        run = mock.Mock()
        monkeypatch.setattr(lsf_instance.subprocess, 'run', run)

        # pylint: disable=protected-access
        assert lsf_instance._terminate_from_inside_allocation(
            'sky-abcd-user') is False
        run.assert_not_called()

    def test_api_server_takes_the_ssh_path(self, monkeypatch):
        monkeypatch.delenv('LSB_JOBID', raising=False)
        run = mock.Mock()
        monkeypatch.setattr(lsf_instance.subprocess, 'run', run)

        # pylint: disable=protected-access
        assert lsf_instance._terminate_from_inside_allocation('sky-x') is False
        run.assert_not_called()


class TestQueueInfo:
    """Queues come from config; live `bqueues` state is enrichment only."""

    def _queues(self):
        return {
            'hpc': lsf_utils.QueueGpuInfo(gpu_type=None, gpu_count=0),
            'gpuv100': lsf_utils.QueueGpuInfo(gpu_type='V100', gpu_count=4),
        }

    def test_configured_shape_with_live_state(self, monkeypatch):
        monkeypatch.setattr(lsf_utils, 'get_configured_queues',
                            lambda cluster: self._queues())
        client = mock.Mock()
        client.get_queues_info.return_value = [
            lsf.LsfQueueInfo(name='gpuv100',
                             priority=50,
                             status='Open:Active',
                             njobs=7,
                             pend=2,
                             run=5),
        ]
        monkeypatch.setattr(lsf_utils, 'make_client', lambda cluster: client)

        rows = lsf_utils.lsf_queue_info('dtu')
        assert [r['queue'] for r in rows] == ['hpc', 'gpuv100']
        # The first configured queue is where unaccelerated work lands.
        assert rows[0]['is_default'] and not rows[1]['is_default']
        assert rows[1]['gpu_type'] == 'V100'
        assert rows[1]['gpu_count_per_host'] == 4
        assert rows[1]['run'] == 5
        # Configured but not reported live: shape without counts.
        assert rows[0]['status'] is None

    def test_live_failure_keeps_the_configured_queues(self, monkeypatch):
        monkeypatch.setattr(lsf_utils, 'get_configured_queues',
                            lambda cluster: self._queues())

        def _boom(cluster):
            raise RuntimeError('login node unreachable')

        monkeypatch.setattr(lsf_utils, 'make_client', _boom)

        rows = lsf_utils.lsf_queue_info('dtu')
        assert [r['queue'] for r in rows] == ['hpc', 'gpuv100']
        assert all(r['status'] is None for r in rows)

    def test_live_false_makes_no_connection(self, monkeypatch):
        monkeypatch.setattr(lsf_utils, 'get_configured_queues',
                            lambda cluster: self._queues())
        make_client = mock.Mock()
        monkeypatch.setattr(lsf_utils, 'make_client', make_client)

        rows = lsf_utils.lsf_queue_info('dtu', live=False)
        assert len(rows) == 2
        make_client.assert_not_called()


class TestNodeInfo:
    """Node rows key on the same names as Slurm's, so the UI renders both."""

    @staticmethod
    def _client():
        client = mock.Mock()
        client.get_hosts.return_value = [
            lsf.HostInfo(host='n-1',
                         status='ok',
                         max_slots=20,
                         njobs=4,
                         run=4),
            lsf.HostInfo(host='n-2',
                         status='closed_Adm',
                         max_slots=20,
                         njobs=0,
                         run=0),
        ]
        client.get_host_resources.return_value = [
            lsf.HostResources(host='n-1',
                              model='IntelXeon',
                              ncpus=20,
                              max_memory_gb=125.8),
        ]
        client.get_host_load.return_value = [
            lsf.LoadInfo(host='n-1',
                         status='ok',
                         cpu_load=1.2,
                         free_memory_gb=100.0),
        ]
        client.get_gpu_hosts.return_value = [
            lsf.GpuHostInfo(host='n-1',
                            gpu_id=0,
                            model='TeslaV100_PCIE_32GB',
                            njobs=0,
                            run=0),
            lsf.GpuHostInfo(host='n-1',
                            gpu_id=1,
                            model='TeslaV100_PCIE_32GB',
                            njobs=1,
                            run=1),
            lsf.GpuHostInfo(host='n-2',
                            gpu_id=0,
                            model='TeslaV100_PCIE_32GB',
                            njobs=0,
                            run=0),
        ]
        return client

    def test_node_rows(self, monkeypatch):
        monkeypatch.setattr(lsf_utils, 'make_client',
                            lambda cluster: self._client())
        nodes = {n['node_name']: n for n in lsf_utils.lsf_node_info('dtu')}

        n1 = nodes['n-1']
        assert n1['lsf_cluster_name'] == 'dtu'
        assert n1['gpu_type'] == 'V100-32GB'
        assert n1['total_gpus'] == 2 and n1['free_gpus'] == 1
        assert n1['vcpu_count'] == 20 and n1['free_vcpus'] == 16
        assert n1['memory_gb'] == 125.8
        assert n1['cpu_load'] == 1.2 and n1['free_memory_gb'] == 100.0

    def test_unusable_host_has_nothing_free(self, monkeypatch):
        monkeypatch.setattr(lsf_utils, 'make_client',
                            lambda cluster: self._client())
        nodes = {n['node_name']: n for n in lsf_utils.lsf_node_info('dtu')}

        # The GPU is idle, but LSF will not dispatch to an administratively
        # closed host, so reporting it as available would be a lie.
        n2 = nodes['n-2']
        assert n2['total_gpus'] == 1 and n2['free_gpus'] == 0

    def test_missing_optional_commands(self, monkeypatch):
        client = self._client()
        client.get_host_resources.return_value = []
        client.get_host_load.return_value = []
        monkeypatch.setattr(lsf_utils, 'make_client', lambda cluster: client)

        n1 = lsf_utils.lsf_node_info('dtu')[0]
        assert n1['vcpu_count'] is None
        assert n1['cpu_load'] is None
        # The roster and the GPU inventory still came through.
        assert n1['total_gpus'] == 2

    def test_explicit_cluster_failure_is_empty_not_raised(self, monkeypatch):

        def _boom(cluster):
            raise RuntimeError('login node unreachable')

        monkeypatch.setattr(lsf_utils, 'make_client', _boom)
        assert lsf_utils.lsf_node_info('dtu') == []
