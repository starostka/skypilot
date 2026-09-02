"""IBM Spectrum LSF."""

import os
import typing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sky import catalog
from sky import clouds
from sky import exceptions
from sky import sky_logging
from sky import skypilot_config
from sky.provision.lsf import creds
from sky.provision.lsf import utils as lsf_utils
from sky.utils import annotations
from sky.utils import common_utils
from sky.utils import config_utils
from sky.utils import registry
from sky.utils import resources_utils

if typing.TYPE_CHECKING:
    from sky import resources as resources_lib
    from sky.utils import volume as volume_lib

logger = sky_logging.init_logger(__name__)

CREDENTIAL_PATH = lsf_utils.DEFAULT_LSF_PATH


@registry.CLOUD_REGISTRY.register
class Lsf(clouds.Cloud):
    """IBM Spectrum LSF."""

    _REPR = 'LSF'
    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.AUTOSTOP: 'LSF does not '
                                                     'support autostop.',
        clouds.CloudImplementationFeatures.AUTODOWN: 'LSF does not '
                                                     'support autodown.',
        clouds.CloudImplementationFeatures.AUTO_TERMINATE: 'LSF does not '
                                                           'support auto-'
                                                           'termination.',
        clouds.CloudImplementationFeatures.STOP: 'LSF does not support '
                                                 'stopping instances.',
        clouds.CloudImplementationFeatures.SPOT_INSTANCE: 'Spot instances are '
                                                          'not supported in '
                                                          'LSF.',
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            'Customized multiple network interfaces are not supported in '
            'LSF.',
        clouds.CloudImplementationFeatures.OPEN_PORTS: 'Opening ports is not '
                                                       'supported in LSF.',
        clouds.CloudImplementationFeatures.HOST_CONTROLLERS:
            'Running '
            'controllers is not '
            'well tested with '
            'LSF.',
        clouds.CloudImplementationFeatures.LOCAL_DISK:
            (f'Local disk is not supported on {_REPR}'),
        clouds.CloudImplementationFeatures.DOCKER_IMAGE:
            'Docker image is not supported on LSF: jobs run directly on '
            'the compute nodes as the submitting user.',
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            'Storage mounting is not supported on LSF: FUSE mounts cannot '
            'be assumed on HPC compute nodes.',
        clouds.CloudImplementationFeatures.MULTI_NODE:
            'The LSF backend currently supports single-node clusters only '
            '(jobs are submitted with span[hosts=1]).',
    }
    _MAX_CLUSTER_NAME_LEN_LIMIT = 120
    _regions: List[clouds.Region] = []
    _INDENT_PREFIX = '    '

    # Same as Kubernetes/Slurm.
    _DEFAULT_NUM_VCPUS_WITH_GPU = 4
    _DEFAULT_MEMORY_CPU_RATIO_WITH_GPU = 4

    # Using the latest SkyPilot provisioner API to provision and check
    # status.
    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        return cls._MAX_CLUSTER_NAME_LEN_LIMIT

    @classmethod
    def optimize_by_zone(cls) -> bool:
        return True

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls,
        instance_type: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        inst = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
        return inst.cpus, inst.memory

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> Iterator[Optional[List[clouds.Zone]]]:
        """Iterate over queues (zones) for provisioning with failover.

        Yields one queue at a time for failover retry logic.
        """
        del num_nodes  # unused

        regions = cls.regions_with_offering(instance_type,
                                            accelerators,
                                            use_spot,
                                            region=region,
                                            zone=None)

        for r in regions:
            if r.zones:
                # Yield one queue at a time for failover.
                for zone in r.zones:
                    yield [zone]
            else:
                # No queues discovered, use default.
                yield None

    @classmethod
    @annotations.lru_cache(scope='global', maxsize=1)
    def _log_skipped_clusters_once(cls, skipped_clusters: Tuple[str,
                                                                ...]) -> None:
        """Log skipped clusters only once."""
        if skipped_clusters:
            logger.warning(
                f'LSF clusters {set(skipped_clusters)!r} specified in '
                '"allowed_clusters" not found in ~/.lsf/config. '
                'Ignoring these clusters.')

    @classmethod
    def existing_allowed_clusters(cls, silent: bool = False) -> List[str]:
        """Get existing allowed clusters.

        Returns clusters based on the following logic:
        1. If 'allowed_clusters' is set to 'all' in ~/.sky/config.yaml,
           return all clusters from ~/.lsf/config
        2. If specific clusters are listed in 'allowed_clusters',
           return only those that exist in ~/.lsf/config
        3. If no configuration is specified, return all clusters
           from ~/.lsf/config (default behavior)
        """
        all_clusters = lsf_utils.get_all_lsf_cluster_names()
        if len(all_clusters) == 0:
            return []

        all_clusters_set = set(all_clusters)

        # Workspace-level allowed_clusters take precedence over the global
        # allowed_clusters.
        allowed_clusters = skypilot_config.get_workspace_cloud('lsf').get(
            'allowed_clusters', None)
        if allowed_clusters is None:
            allowed_clusters = skypilot_config.get_effective_region_config(
                cloud='lsf',
                region=None,
                keys=('allowed_clusters',),
                default_value=None)

        allow_all_clusters = allowed_clusters == 'all'
        if allow_all_clusters or allowed_clusters is None:
            allowed_clusters = list(all_clusters)

        existing_clusters = []
        skipped_clusters = []
        for cluster in allowed_clusters:
            if cluster in all_clusters_set:
                existing_clusters.append(cluster)
            else:
                skipped_clusters.append(cluster)

        if not silent:
            cls._log_skipped_clusters_once(tuple(sorted(skipped_clusters)))

        return existing_clusters

    @classmethod
    def regions_with_offering(
        cls,
        instance_type: Optional[str],
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
        resources: Optional['resources_lib.Resources'] = None
    ) -> List[clouds.Region]:
        del accelerators, use_spot, resources  # unused
        existing_clusters = cls.existing_allowed_clusters()

        regions: List[clouds.Region] = []
        for cluster in existing_clusters:
            # Filter by region if specified.
            if region is not None and cluster != region:
                continue

            # Fetch queues for this cluster and attach as zones.
            try:
                queues = lsf_utils.get_queues(cluster)
                if zone is not None:
                    # Filter by zone (queue) if specified.
                    queues = [q for q in queues if q == zone]
                zones = [clouds.Zone(q) for q in queues]
            except Exception as e:  # pylint: disable=broad-except
                logger.debug(f'Failed to get queues for {cluster}: {e}')
                zones = []

            r = clouds.Region(cluster)
            if zones:
                r.set_zones(zones)
            regions.append(r)

        # Check if the requested instance type fits in each cluster/queue.
        if instance_type is None:
            return regions

        regions_to_return = []
        for r in regions:
            cluster = r.name
            queues_to_check = [z.name for z in r.zones] if r.zones else []
            valid_zones = []

            # Narrow the queue list based on the configured queue map:
            # GPU jobs go to queues that declare the GPU type, CPU jobs to
            # queues without one.
            try:
                inst = lsf_utils.LsfInstanceType.from_instance_type(
                    instance_type)
                if inst.accelerator_type is not None:
                    acc_count = inst.accelerator_count or 0
                    matching = lsf_utils.queues_for_accelerator(
                        cluster, inst.accelerator_type, acc_count)
                    available = set(queues_to_check)
                    queues_to_check = [q for q in matching if q in available]
                else:
                    available = set(queues_to_check)
                    queues_to_check = [
                        q for q in lsf_utils.cpu_queues(cluster)
                        if q in available
                    ]
            except ValueError:
                pass

            for queue in queues_to_check:
                fits, reason = lsf_utils.check_instance_fits(
                    cluster, instance_type, queue)
                if fits:
                    valid_zones.append(clouds.Zone(queue))
                else:
                    logger.debug(
                        f'Instance type {instance_type} does not fit in '
                        f'{cluster}/{queue}: {reason}')

            if valid_zones:
                r.set_zones(valid_zones)
                regions_to_return.append(r)

        return regions_to_return

    def instance_type_to_hourly_cost(self,
                                     instance_type: str,
                                     use_spot: bool,
                                     region: Optional[str] = None,
                                     zone: Optional[str] = None) -> float:
        # pylint: disable=import-outside-toplevel
        from sky.catalog import lsf_catalog
        return lsf_catalog.get_hourly_cost(instance_type, use_spot, region,
                                           zone)

    def accelerators_to_hourly_cost(self,
                                    accelerators: Dict[str, int],
                                    use_spot: bool,
                                    region: Optional[str] = None,
                                    zone: Optional[str] = None) -> float:
        """Returns the hourly cost of the accelerators, in dollars/hour."""
        del accelerators, use_spot, region, zone  # unused
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        return 0.0

    def __repr__(self):
        return self._REPR

    def is_same_cloud(self, other: clouds.Cloud) -> bool:
        # Returns true if the two clouds are the same cloud type.
        return isinstance(other, Lsf)

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
    ) -> Optional[str]:
        """Returns the default instance type for LSF."""
        del max_hourly_cost  # Unused.
        return catalog.get_default_instance_type(cpus=cpus,
                                                 memory=memory,
                                                 disk_tier=disk_tier,
                                                 local_disk=local_disk,
                                                 region=region,
                                                 zone=zone,
                                                 use_spot=use_spot,
                                                 clouds='lsf')

    @classmethod
    def get_accelerators_from_instance_type(
            cls, instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
        inst = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
        return {
            inst.accelerator_type: inst.accelerator_count
        } if (inst.accelerator_count is not None and
              inst.accelerator_type is not None) else None

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def make_deploy_resources_variables(
        self,
        resources: 'resources_lib.Resources',
        cluster_name: 'resources_utils.ClusterName',
        region: Optional['clouds.Region'],
        zones: Optional[List['clouds.Zone']],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Any]:
        del cluster_name, dryrun, volume_mounts  # Unused.
        del num_nodes  # Single-node only; enforced via MULTI_NODE feature.
        assert region is not None, 'No available LSF cluster found.'
        cluster = region.name

        resources = resources.assert_launchable()
        acc_dict = self.get_accelerators_from_instance_type(
            resources.instance_type)
        custom_resources = resources_utils.make_ray_custom_resources_str(
            acc_dict)

        inst = lsf_utils.LsfInstanceType.from_instance_type(
            resources.instance_type)
        cpus = inst.cpus
        mem = inst.memory
        acc_count = inst.accelerator_count if inst.accelerator_count else 0
        acc_type = inst.accelerator_type if inst.accelerator_type else None

        # Use zone as queue if specified, otherwise pick from the queue
        # map (GPU type -> queue) or the CPU queues.
        if zones and len(zones) > 0:
            queue = zones[0].name
        else:
            if acc_type is not None:
                candidates = lsf_utils.queues_for_accelerator(
                    cluster, acc_type, acc_count)
            else:
                candidates = lsf_utils.cpu_queues(cluster)
            if not candidates:
                raise exceptions.ResourcesUnavailableError(
                    f'No suitable LSF queue found on cluster {cluster!r} '
                    f'for instance type {resources.instance_type!r}. '
                    'Declare queues (and their GPU types) under '
                    f'lsf.cluster_configs.{cluster}.queues in '
                    '~/.sky/config.yaml.')
            queue = candidates[0]

        credentials = lsf_utils.get_lsf_credentials(cluster)

        provision_timeout = skypilot_config.get_effective_region_config(
            cloud='lsf',
            region=cluster,
            keys=('provision_timeout',),
            default_value=None)
        if provision_timeout is None:
            if resources.zone is not None:
                # When the queue is pinned there is no failover, so let LSF
                # hold on to the job in the queue.
                provision_timeout = 24 * 60 * 60  # 24 hours
            else:
                provision_timeout = 2 * 60  # 2 minutes

        # Read bsub_options with three-level merge:
        # global < cluster < queue.
        bsub_options: Dict[str, Any] = {}
        for config_keys in [
            ('lsf', 'bsub_options'),
            ('lsf', 'cluster_configs', cluster, 'bsub_options'),
            ('lsf', 'cluster_configs', cluster, 'queue_configs', queue,
             'bsub_options'),
        ]:
            level_config = skypilot_config.get_nested(config_keys,
                                                      default_value=None)
            if level_config is not None:
                bsub_options.update(level_config)
        # Merge task-level config overrides (from `config:` in task YAML).
        task_bsub = config_utils.get_cloud_config_value_from_dict(
            dict_config=resources.cluster_config_overrides,
            cloud='lsf',
            region=cluster,
            keys=('bsub_options',))
        if task_bsub is not None:
            bsub_options.update(task_bsub)

        walltime = lsf_utils.get_default_walltime(cluster)

        # ProxyCommand through the login node, used in the generated
        # cluster config's auth section so `ssh <cluster>` (and any
        # credential-based SSH) can reach the reverse-tunneled sshd.
        login_proxy_command = lsf_utils.build_login_proxy_command({
            'hostname': credentials.host,
            'port': credentials.port,
            'user': credentials.user,
            'private_key': credentials.identity_file,
            'proxycommand': credentials.proxy_command,
            'proxyjump': credentials.proxy_jump,
        })

        deploy_vars = {
            'instance_type': resources.instance_type,
            'custom_resources': custom_resources,
            'cpus': str(cpus),
            'memory': str(mem),
            'accelerator_count': str(acc_count),
            'accelerator_type': acc_type,
            'lsf_cluster': cluster,
            'lsf_queue': queue,
            'provision_timeout': provision_timeout,
            'walltime': walltime,
            'ssh_hostname': credentials.host,
            'ssh_port': str(credentials.port),
            'ssh_user': credentials.user,
            'lsf_proxy_command': credentials.proxy_command,
            'lsf_proxy_jump': credentials.proxy_jump,
            'lsf_identities_only': credentials.identities_only,
            # NOTE: named lsf_private_key to avoid colliding with the
            # SkyPilot key ('ssh_private_key' in the auth section); see the
            # 'ssh' and 'auth' sections of lsf-ray.yml.j2.
            'lsf_private_key': credentials.identity_file,
            'lsf_login_proxy_command': login_proxy_command,
            'bsub_options': bsub_options,
        }

        return deploy_vars

    def _get_feasible_launchable_resources(
        self, resources: 'resources_lib.Resources'
    ) -> 'resources_utils.FeasibleResources':
        """Returns a list of feasible resources for the given resources."""
        if resources.instance_type is not None:
            assert resources.is_launchable(), resources
            # Check if the instance type is available in at least one
            # cluster.
            available_regions = self.regions_with_offering(
                resources.instance_type,
                accelerators=None,
                use_spot=resources.use_spot,
                region=resources.region,
                zone=resources.zone,
                resources=resources)
            if not available_regions:
                return resources_utils.FeasibleResources([], [], None)

            # Return a single resource without region set. The optimizer
            # will call make_launchables_for_valid_region_zones() which
            # creates one resource per region/cluster.
            resources = resources.copy(accelerators=None)
            return resources_utils.FeasibleResources([resources], [], None)

        def _make(instance_list):
            resource_list = []
            for instance_type in instance_list:
                r = resources.copy(
                    cloud=Lsf(),
                    instance_type=instance_type,
                    accelerators=None,
                )
                resource_list.append(r)
            return resource_list

        # Currently, handle a filter on accelerators only.
        accelerators = resources.accelerators

        default_instance_type = Lsf.get_default_instance_type(
            cpus=resources.cpus,
            memory=resources.memory,
            disk_tier=resources.disk_tier,
            local_disk=resources.local_disk,
            region=resources.region,
            zone=resources.zone,
            use_spot=resources.use_spot,
            max_hourly_cost=resources.max_hourly_cost)
        if default_instance_type is None:
            return resources_utils.FeasibleResources([], [], None)

        if accelerators is None:
            chosen_instance_type = default_instance_type
        else:
            assert len(accelerators) == 1, resources

            # Build a GPU-enabled instance type.
            acc_type, acc_count = list(accelerators.items())[0]

            lsf_instance_type = (lsf_utils.LsfInstanceType.from_instance_type(
                default_instance_type))

            gpu_task_cpus = lsf_instance_type.cpus
            if resources.cpus is None:
                gpu_task_cpus = self._DEFAULT_NUM_VCPUS_WITH_GPU * acc_count
            if resources.memory is not None:
                gpu_task_memory = float(resources.memory.strip('+'))
            else:
                gpu_task_memory = (gpu_task_cpus *
                                   self._DEFAULT_MEMORY_CPU_RATIO_WITH_GPU)

            chosen_instance_type = (lsf_utils.LsfInstanceType.from_resources(
                gpu_task_cpus, gpu_task_memory, acc_count, acc_type).name)

        # Check the availability of the chosen instance type in all LSF
        # clusters.
        available_regions = self.regions_with_offering(
            chosen_instance_type,
            accelerators=None,
            use_spot=resources.use_spot,
            region=resources.region,
            zone=resources.zone,
            resources=resources)
        if not available_regions:
            return resources_utils.FeasibleResources([], [], None)

        return resources_utils.FeasibleResources(_make([chosen_instance_type]),
                                                 [], None)

    @classmethod
    def _check_compute_credentials(
            cls) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        """Checks if the user has access credentials to the LSF cluster."""
        all_clusters = []
        try:
            all_clusters = lsf_utils.get_all_lsf_cluster_names()
        except Exception as e:  # pylint: disable=broad-except
            return (False, 'Failed to load LSF configuration from '
                    f'{lsf_utils.DEFAULT_LSF_PATH}: '
                    f'{common_utils.format_exception(e)}.')
        if not all_clusters:
            return (False,
                    f'LSF configuration file {lsf_utils.DEFAULT_LSF_PATH} does '
                    'not exist or contains no clusters.\n'
                    f'{cls._INDENT_PREFIX}Configure at least one LSF cluster '
                    'in SSH config format (Host/HostName/User/IdentityFile).')

        existing_allowed_clusters = cls.existing_allowed_clusters()
        if not existing_allowed_clusters:
            return (False, 'No allowed LSF clusters found in '
                    f'{lsf_utils.DEFAULT_LSF_PATH}.')

        # A DEPLOYMENT THAT BROKERS PER-USER CREDENTIALS CANNOT BE PROBED HERE.
        # The loop below opens a connection AS THE SERVER, which presumes a
        # shared account the server can log in with. Where credentials are
        # issued per caller (creds.CREDENTIAL_PROVIDER_ENV_VAR), no such
        # account exists and there is no caller during `sky check`, so the
        # probe fails for a deployment that is correctly configured — and the
        # cloud is then reported disabled and every launch refused.
        #
        # Validate what CAN be checked without an identity: the config names
        # clusters, and those clusters are allowed. Whether a given user may
        # actually log in is answered at request time, by the store, against
        # that user's own token — which is the only place it can be answered.
        if os.environ.get(creds.CREDENTIAL_PROVIDER_ENV_VAR):
            return True, {
                cluster: 'enabled (per-user credentials; not probed)'
                for cluster in existing_allowed_clusters
            }

        # Check credentials for each cluster and return a ctx2text mapping.
        ctx2text = {}
        success = False
        for cluster in existing_allowed_clusters:
            try:
                client = lsf_utils.make_client(cluster)
                info = client.check_reachable()
                logger.debug(f'LSF cluster {cluster} lsid: {info}')
                ctx2text[cluster] = 'enabled'
                success = True
            except KeyError as e:
                key = e.args[0]
                ctx2text[cluster] = (
                    f'disabled. {str(key).capitalize()} is missing, please '
                    'check your ~/.lsf/config and try again.')
            except Exception as e:  # pylint: disable=broad-except
                error_msg = (f'Credential check failed: '
                             f'{common_utils.format_exception(e)}')
                ctx2text[cluster] = f'disabled. {error_msg}'

        return success, ctx2text

    def get_credential_file_mounts(self) -> Dict[str, str]:
        # LSF control-plane credentials never leave the API server: jobs on
        # the cluster do not need them.
        return {}

    @classmethod
    def get_current_user_identity(cls) -> Optional[List[str]]:
        return None

    def instance_type_exists(self, instance_type: str) -> bool:
        return catalog.instance_type_exists(instance_type, 'lsf')

    def validate_region_zone(self, region: Optional[str], zone: Optional[str]):
        """Validate region (cluster) and zone (queue).

        Args:
            region: LSF cluster alias.
            zone: LSF queue name (optional).

        Returns:
            Tuple of (region, zone) if valid.

        Raises:
            ValueError: If the cluster or queue is not found.
        """
        all_clusters = lsf_utils.get_all_lsf_cluster_names()
        if region and region not in all_clusters:
            raise ValueError(
                f'Cluster {region} not found in LSF config. LSF only '
                'supports cluster aliases as regions. Available '
                f'clusters: {all_clusters}')

        # Validate queue (zone) if specified.
        if zone is not None:
            if region is None:
                raise ValueError(
                    'Cannot specify queue (zone) without specifying '
                    'cluster (region) for LSF.')

            queues = lsf_utils.get_queues(region)
            if zone not in queues:
                raise ValueError(
                    f'Queue {zone!r} not found in cluster {region!r}. '
                    f'Available queues: {queues}')

        return region, zone

    def accelerator_in_region_or_zone(self,
                                      accelerator: str,
                                      acc_count: int,
                                      region: Optional[str] = None,
                                      zone: Optional[str] = None) -> bool:
        del zone  # unused for now
        regions = catalog.get_region_zones_for_accelerators(accelerator,
                                                            acc_count,
                                                            use_spot=False,
                                                            clouds='lsf')
        if not regions:
            return False
        if region is None:
            return True
        return any(r.name == region for r in regions)

    @classmethod
    def expand_infras(cls) -> List[str]:
        """Returns a list of enabled LSF clusters.

        Each is returned as 'LSF/cluster-name'.
        """
        infras = []
        for cluster in cls.existing_allowed_clusters(silent=True):
            infras.append(f'{cls.canonical_name()}/{cluster}')
        return infras
