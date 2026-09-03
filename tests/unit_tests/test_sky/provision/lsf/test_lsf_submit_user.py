"""Per-caller identity for LSF submission.

These cover the property the feature exists for: two different callers must
resolve to two different Unix accounts, and a caller whose identity cannot be
turned into a valid account must fail loudly rather than fall back to whatever
account the credential source happens to name.
"""
from unittest import mock

import pytest

from sky.provision.lsf import utils as lsf_utils
from sky.utils import schemas


def _config(enabled):
    """Patches the submit_as_user lookup."""
    return mock.patch.object(lsf_utils.skypilot_config,
                             'get_effective_region_config',
                             return_value=enabled)


def _caller(name):
    return mock.patch.object(lsf_utils.common_utils,
                             'get_current_user_name',
                             return_value=name)


def test_disabled_returns_none():
    """Off by default: the credential source keeps naming the account."""
    with _config(False), _caller('alice@example.edu'):
        assert lsf_utils.get_submit_user('mycluster') is None


def test_derives_local_part_of_sso_identity():
    with _config(True), _caller('alice@example.edu'):
        assert lsf_utils.get_submit_user('mycluster') == 'alice'


def test_bare_username_passes_through():
    with _config(True), _caller('alice'):
        assert lsf_utils.get_submit_user('mycluster') == 'alice'


def test_account_with_digits():
    """Many sites issue numeric-suffixed accounts."""
    with _config(True), _caller('s123456@example.edu'):
        assert lsf_utils.get_submit_user('mycluster') == 's123456'


def test_two_callers_resolve_differently():
    """The whole point: identity is per call, not per process."""
    with _config(True):
        with _caller('alice@example.edu'):
            first = lsf_utils.get_submit_user('mycluster')
        with _caller('bob@example.edu'):
            second = lsf_utils.get_submit_user('mycluster')
    assert (first, second) == ('alice', 'bob')


@pytest.mark.parametrize('name', [
    'Alice@example.edu',
    '1alice@example.edu',
    'al ice@example.edu',
    'alice;rm -rf@example.edu',
    '../../etc/passwd@example.edu',
    '@example.edu',
])
def test_invalid_identities_raise(name):
    """Must raise, never silently degrade.

    A returned None here would mean "use the credential source's account",
    i.e. an unmappable identity would submit as somebody else. These inputs
    also cover the shell-injection shapes, since the value reaches an ssh
    command line.
    """
    with _config(True), _caller(name):
        with pytest.raises(ValueError, match='valid Unix user'):
            lsf_utils.get_submit_user('mycluster')


def test_flag_is_declared_in_the_schema():
    """The config key must be accepted, at both levels.

    Guards a mistake that shipped once: the implementation read
    submit_as_user while only the SLURM schema declared it. LSF sets
    additionalProperties: False, so any config using the flag was rejected
    at validation and the feature could not be turned on at all — while the
    code around it looked complete.
    """
    lsf = schemas.get_config_schema()['properties']['lsf']
    assert 'submit_as_user' in lsf['properties']
    per_cluster = lsf['properties']['cluster_configs']['additionalProperties']
    assert 'submit_as_user' in per_cluster['properties']


def test_a_config_using_the_flag_validates():
    """End to end: the shape a deployment actually writes must pass."""
    import jsonschema
    jsonschema.validate(
        {'lsf': {'cluster_configs': {'mycluster': {'submit_as_user': True}}}},
        schemas.get_config_schema())


def test_optimizer_can_evaluate_lsf_resources():
    """The optimizer must be able to ask LSF what it does not support.

    check_features_are_supported() calls _unsupported_features_for_resources()
    for every candidate. Leaving it to the base class raises a bare
    NotImplementedError from inside the optimizer, and a launch then fails with
    an EMPTY message — no cloud named, no feature named, nothing to act on.
    """
    import sky
    from sky.clouds import lsf as lsf_cloud

    r = sky.Resources(infra='lsf/dtu', cpus='1')
    unsupported = lsf_cloud.Lsf._unsupported_features_for_resources(r)

    assert isinstance(unsupported, dict)
    # A property of the backend, not of a cluster: no connection is needed to
    # answer, which is why this is returned statically.
    assert lsf_cloud.clouds.CloudImplementationFeatures.STOP in unsupported
