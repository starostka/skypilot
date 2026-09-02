"""Credential resolution for LSF login nodes.

The LSF backend reaches the cluster's login node over SSH for every
control-plane operation. This module isolates *where those SSH credentials
come from* behind a small provider interface, so that a platform deployment
can plug in a dynamic source (e.g. short-lived certificates issued by a
secret manager) without touching the rest of the backend.

The default provider mirrors the Slurm backend: a static SSH-config-format
file at ``~/.lsf/config``, where each ``Host`` alias names one LSF cluster:

    Host mycluster
        HostName login1.hpc.example.edu
        User alice
        IdentityFile ~/.ssh/id_ed25519
"""

import os
import typing
from typing import Any, Dict, List, NamedTuple, Optional

if typing.TYPE_CHECKING:
    from paramiko.config import SSHConfig

DEFAULT_LSF_PATH = '~/.lsf/config'


class LsfCredentials(NamedTuple):
    """SSH credentials for an LSF login node."""
    host: str
    port: int
    user: str
    # Path to the SSH private key, or None for ssh-agent/keyless auth.
    identity_file: Optional[str]
    # Path to an SSH certificate, for deployments using a CA-based flow
    # (e.g. keys signed by an SSH CA). Optional and currently only
    # carried through; the default OpenSSH client picks up `<key>-cert.pub`
    # automatically.
    cert_file: Optional[str]
    proxy_command: Optional[str]
    proxy_jump: Optional[str]
    identities_only: bool


class CredentialProvider:
    """Interface for resolving LSF login-node credentials.

    Implementations must be cheap to call; results may be requested
    frequently (each control-plane operation resolves credentials).
    """

    def list_clusters(self) -> List[str]:
        """Returns all configured LSF cluster aliases."""
        raise NotImplementedError

    def get_credentials(self, cluster: str) -> LsfCredentials:
        """Returns login-node SSH credentials for the given cluster alias.

        Raises:
            KeyError: If required fields (e.g. User) are missing.
            FileNotFoundError: If the credential source does not exist.
        """
        raise NotImplementedError


class SSHConfigCredentialProvider(CredentialProvider):
    """Default provider reading a static SSH config file (~/.lsf/config)."""

    def __init__(self, path: str = DEFAULT_LSF_PATH):
        self._path = path

    def _load(self) -> 'SSHConfig':
        # Import lazily so that merely importing the LSF backend does not
        # require paramiko.
        # pylint: disable-next=import-outside-toplevel
        from paramiko.config import SSHConfig
        path = os.path.expanduser(self._path)
        return SSHConfig.from_path(path)

    def list_clusters(self) -> List[str]:
        try:
            ssh_config = self._load()
        except FileNotFoundError:
            return []
        return [h for h in ssh_config.get_hostnames() if h != '*']

    def get_credentials(self, cluster: str) -> LsfCredentials:
        ssh_config = self._load()
        config_dict: Dict[str, Any] = ssh_config.lookup(cluster)
        identity_files = config_dict.get('identityfile')
        identity_file = identity_files[0] if identity_files else None
        cert_files = config_dict.get('certificatefile')
        cert_file = cert_files[0] if cert_files else None
        identities_only = (config_dict.get('identitiesonly',
                                           '').lower() == 'yes')
        return LsfCredentials(
            host=config_dict['hostname'],
            port=int(config_dict.get('port', 22)),
            user=config_dict['user'],
            identity_file=identity_file,
            cert_file=cert_file,
            proxy_command=config_dict.get('proxycommand', None),
            proxy_jump=config_dict.get('proxyjump', None),
            identities_only=identities_only,
        )


# A deployment that brokers credentials from a secret store (short-lived SSH
# certificates, per-user keys) plugs in here instead of vendoring its
# integration into this backend. Point it at a factory:
#
#   SKYPILOT_LSF_CREDENTIAL_PROVIDER=my_platform.lsf_creds:make_provider
#
# The factory takes no arguments and returns a CredentialProvider. It is
# resolved once, on first use, so an import error surfaces at the first LSF
# operation rather than at interpreter start.
CREDENTIAL_PROVIDER_ENV_VAR = 'SKYPILOT_LSF_CREDENTIAL_PROVIDER'

_provider: Optional[CredentialProvider] = None
_provider_explicitly_set = False


def _load_provider_from_env() -> Optional[CredentialProvider]:
    spec = os.environ.get(CREDENTIAL_PROVIDER_ENV_VAR, '').strip()
    if not spec:
        return None
    if ':' not in spec:
        raise ValueError(
            f'{CREDENTIAL_PROVIDER_ENV_VAR} must be "module:callable", '
            f'got {spec!r}.')
    module_name, _, attr = spec.partition(':')
    # pylint: disable-next=import-outside-toplevel
    import importlib
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    provider = factory()
    if not isinstance(provider, CredentialProvider):
        raise TypeError(
            f'{spec} returned {type(provider).__name__}, which is not a '
            'CredentialProvider.')
    return provider


def get_provider() -> CredentialProvider:
    """Returns the active credential provider."""
    global _provider
    if _provider is None:
        # An explicit set_provider() always wins over the environment.
        _provider = (_load_provider_from_env() or
                     SSHConfigCredentialProvider())
    return _provider


def set_provider(provider: CredentialProvider) -> None:
    """Replaces the active credential provider (e.g. for a platform plugin)."""
    global _provider, _provider_explicitly_set
    _provider = provider
    _provider_explicitly_set = True
