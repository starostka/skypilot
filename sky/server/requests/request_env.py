"""Per-request environment contributed by a deployment plugin.

Requests do not execute in the API server process. `prepare_request_async`
serialises identity into `request_body.env_vars`, which is the only channel
that reaches the worker. Anything a backend needs that exists *only* in the
server's request context — most obviously a credential brokered on the
caller's behalf — has to travel the same way.

A deployment names a callable::

    SKYPILOT_REQUEST_ENV_HOOK=my_platform.creds:contribute_env

It receives the caller's access token (or None) and returns a
``Dict[str, str]`` merged into the request's env_vars.

The token is passed EXPLICITLY rather than read from ambient state. The
obvious alternative — stashing it in a context variable from middleware —
is unsafe here: sky.utils.context falls back to a PROCESS-GLOBAL dict when
no context is active, and the server initialises a context inside request
handlers rather than in middleware. One caller's credential would then be
readable by another caller's request.

Two deliberate choices:

* **Failures propagate.** A hook that brokers a credential and quietly
  returns nothing produces a job that runs with the wrong identity, or none,
  and fails much later somewhere unrelated. Refusing the request is the
  cheaper failure.
* **What the hook returns is persisted.** Requests are stored, so a hook must
  contribute something it is willing to have at rest — a single-use,
  short-TTL handle rather than a bearer token. That constraint belongs to the
  hook, not here, but it is the reason this interface passes strings by value
  instead of holding a live credential.
"""
import importlib
import os
from typing import Callable, Dict, Optional

from sky import sky_logging

logger = sky_logging.init_logger(__name__)

REQUEST_ENV_HOOK_ENV_VAR = 'SKYPILOT_REQUEST_ENV_HOOK'

_hook: Optional[Callable[[Optional[str]], Dict[str, str]]] = None
_resolved = False


def _resolve() -> Optional[Callable[[Optional[str]], Dict[str, str]]]:
    global _hook, _resolved
    if _resolved:
        return _hook
    _resolved = True
    spec = os.environ.get(REQUEST_ENV_HOOK_ENV_VAR, '').strip()
    if not spec:
        return None
    if ':' not in spec:
        raise ValueError(f'{REQUEST_ENV_HOOK_ENV_VAR} must be '
                         f'"module:callable", got {spec!r}.')
    module_name, _, attr = spec.partition(':')
    module = importlib.import_module(module_name)
    _hook = getattr(module, attr)
    if not callable(_hook):
        raise TypeError(f'{spec} is not callable.')
    logger.info('Per-request env hook: %s', spec)
    return _hook


def contribute(env_vars: Dict[str, str],
               access_token: Optional[str] = None) -> None:
    """Merges the plugin's contribution into a request's env_vars, in place.

    Args:
        env_vars: the request's env_vars, mutated in place.
        access_token: the CALLER's access token, or None when the request
            carries no user credential. A hook that brokers a downstream
            credential must present this rather than a service token: a secret
            store binds what it issues to the presenting identity, so a service
            token would let the server obtain a credential for anyone.
    """
    hook = _resolve()
    if hook is None:
        return
    # THE EXACT JUNCTION. A hook that is configured and a caller that is
    # authenticated can still yield no credential, and until now nothing said
    # which side was empty: the middleware logs only when it captures nothing,
    # and the hook returns {} silently for an absent token. Both look identical
    # downstream — a backend refusing for want of a credential.
    #
    # Length, never the value: this is a bearer token.
    logger.warning('request env hook: token %s',
                   f'present (len {len(access_token)})' if access_token
                   else repr(access_token))
    extra = hook(access_token)
    logger.warning('request env hook contributed: %s', sorted(extra or {}))
    if not extra:
        return
    for key, value in extra.items():
        # Never log values: a hook exists precisely to carry credential
        # material across the process boundary.
        env_vars[str(key)] = str(value)


def reset_for_testing() -> None:
    """Clears the memoised hook so a test can change the environment."""
    global _hook, _resolved
    _hook = None
    _resolved = False
