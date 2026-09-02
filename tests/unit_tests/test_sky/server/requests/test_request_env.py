"""The per-request env contribution hook."""
import os
from unittest import mock

import pytest

from sky.server.requests import request_env


def contribute_env(access_token=None):
    # Echo the token back so a test can prove the CALLER's token is what
    # reaches the hook, not some ambient value.
    return {'MY_HANDLE': 'abc123', 'SAW_TOKEN': access_token or ''}


def contribute_nothing(access_token=None):
    return {}


def explode(access_token=None):
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
        request_env.contribute(env, 'tok-alice')
    assert env == {
        'USER': 'alice',
        'MY_HANDLE': 'abc123',
        'SAW_TOKEN': 'tok-alice',
    }


def test_each_call_sees_its_own_caller_token():
    """The property the explicit argument exists for.

    The token used to be read from a context variable. Nothing ever set it, so
    the hook always saw None — and setting it from middleware would have been
    worse: sky.utils.context falls back to a PROCESS-GLOBAL dict when no
    context is active, so one caller's credential would be visible to another
    caller's request. Passing it per call makes that impossible to reintroduce.
    """
    first, second = {}, {}
    with _env(f'{__name__}:contribute_env'):
        request_env.contribute(first, 'tok-alice')
        request_env.contribute(second, 'tok-bob')
    assert (first['SAW_TOKEN'], second['SAW_TOKEN']) == ('tok-alice', 'tok-bob')


def test_absent_token_is_passed_through_as_none():
    """An unauthenticated request must reach the hook as None, not as a
    stale value left over from whoever called last."""
    env = {}
    with _env(f'{__name__}:contribute_env'):
        request_env.contribute(env, 'tok-alice')
        request_env.contribute(env, None)
    assert env['SAW_TOKEN'] == ''


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
