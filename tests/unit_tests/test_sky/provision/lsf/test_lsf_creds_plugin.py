"""The credential-provider plugin hook.

A deployment that brokers credentials from a secret store plugs in here, so
that integration never has to be vendored into this backend.
"""
import os
from unittest import mock

import pytest

from sky.provision.lsf import creds


class _Stub(creds.CredentialProvider):

    def list_clusters(self):
        return ['stub']

    def get_credentials(self, cluster):
        raise NotImplementedError


def make_provider():
    return _Stub()


def not_a_provider():
    return object()


@pytest.fixture(autouse=True)
def _reset():
    """Provider resolution is cached, so each test starts from unset."""
    creds._provider = None  # pylint: disable=protected-access
    yield
    creds._provider = None  # pylint: disable=protected-access


def test_defaults_to_ssh_config():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(creds.CREDENTIAL_PROVIDER_ENV_VAR, None)
        assert isinstance(creds.get_provider(),
                          creds.SSHConfigCredentialProvider)


def test_loads_provider_from_env():
    spec = f'{__name__}:make_provider'
    with mock.patch.dict(os.environ,
                         {creds.CREDENTIAL_PROVIDER_ENV_VAR: spec}):
        assert isinstance(creds.get_provider(), _Stub)


def test_set_provider_beats_env():
    """An explicit call must win, or tests and embedders cannot override."""
    spec = f'{__name__}:make_provider'
    with mock.patch.dict(os.environ,
                         {creds.CREDENTIAL_PROVIDER_ENV_VAR: spec}):
        sentinel = _Stub()
        creds.set_provider(sentinel)
        assert creds.get_provider() is sentinel


def test_malformed_spec_raises():
    with mock.patch.dict(os.environ,
                         {creds.CREDENTIAL_PROVIDER_ENV_VAR: 'no_colon_here'}):
        with pytest.raises(ValueError, match='module:callable'):
            creds.get_provider()


def test_factory_returning_wrong_type_raises():
    """Fail at load, not at the first credential lookup mid-provision."""
    spec = f'{__name__}:not_a_provider'
    with mock.patch.dict(os.environ,
                         {creds.CREDENTIAL_PROVIDER_ENV_VAR: spec}):
        with pytest.raises(TypeError, match='not a CredentialProvider'):
            creds.get_provider()


def test_check_skips_the_probe_when_credentials_are_per_user():
    """`sky check` must not require the SERVER to be able to log in.

    The probe opens a connection as the server, which presumes a shared
    account. Where a credential provider issues per-caller credentials no such
    account exists, and there is no caller during `sky check` — so a correctly
    configured deployment would be reported disabled and every launch refused.
    """
    from sky.clouds import lsf as lsf_cloud

    with mock.patch.object(lsf_cloud.lsf_utils, 'get_all_lsf_cluster_names',
                           return_value=['dtu']), \
         mock.patch.object(lsf_cloud.Lsf, 'existing_allowed_clusters',
                           classmethod(lambda cls: ['dtu'])), \
         mock.patch.object(lsf_cloud.lsf_utils, 'make_client') as probe, \
         mock.patch.dict(os.environ,
                         {creds.CREDENTIAL_PROVIDER_ENV_VAR:
                          f'{__name__}:make_provider'}):
        ok, detail = lsf_cloud.Lsf._check_compute_credentials()

    assert ok is True
    assert 'dtu' in detail
    # The point: no connection was attempted.
    probe.assert_not_called()
