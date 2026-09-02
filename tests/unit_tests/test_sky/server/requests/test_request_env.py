"""The per-request env contribution hook."""
import os
from unittest import mock

import pytest

from sky.server.requests import request_env


def contribute_env():
    return {'MY_HANDLE': 'abc123'}


def contribute_nothing():
    return {}


def explode():
    raise RuntimeError('broker unavailable')


not_callable = 'definitely not a function'


@pytest.fixture(autouse=True)
def _reset():
    request_env.reset_for_testing()
    yield
    request_env.reset_for_testing()


def _env(spec):
    return mock.patch.dict(os.environ,
                           {request_env.REQUEST_ENV_HOOK_ENV_VAR: spec})


def test_no_hook_is_a_noop():
    env = {'USER': 'alice'}
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(request_env.REQUEST_ENV_HOOK_ENV_VAR, None)
        request_env.contribute(env)
    assert env == {'USER': 'alice'}


def test_hook_contribution_is_merged():
    env = {'USER': 'alice'}
    with _env(f'{__name__}:contribute_env'):
        request_env.contribute(env)
    assert env == {'USER': 'alice', 'MY_HANDLE': 'abc123'}


def test_empty_contribution_is_fine():
    env = {'USER': 'alice'}
    with _env(f'{__name__}:contribute_nothing'):
        request_env.contribute(env)
    assert env == {'USER': 'alice'}


def test_hook_failure_propagates():
    """A broker that fails must fail the request, not the job much later.

    Swallowing this produces a job that runs with no credential and dies
    somewhere unrelated, which is far more expensive to diagnose.
    """
    with _env(f'{__name__}:explode'):
        with pytest.raises(RuntimeError, match='broker unavailable'):
            request_env.contribute({})


def test_malformed_spec_raises():
    with _env('no_colon'):
        with pytest.raises(ValueError, match='module:callable'):
            request_env.contribute({})


def test_non_callable_raises():
    with _env(f'{__name__}:not_callable'):
        with pytest.raises(TypeError, match='not callable'):
            request_env.contribute({})
