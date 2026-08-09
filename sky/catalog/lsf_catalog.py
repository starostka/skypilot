"""LSF Catalog.

The queue -> GPU map is declared statically in the SkyPilot config
(``lsf.cluster_configs.<cluster>.queues``) because LSF selects the GPU
model by queue and does not expose a queue-level GPU inventory. Real-time
capacity/availability is enriched live from ``bhosts -gpu -w``.
"""

import collections
import re
from typing import Dict, List, Optional, Set, Tuple

from sky import check as sky_check
from sky import clouds as sky_clouds
from sky import sky_logging
from sky.catalog import common
from sky.clouds import cloud
from sky.provision.lsf import utils as lsf_utils
from sky.utils import resources_utils

logger = sky_logging.init_logger(__name__)

_DEFAULT_NUM_VCPUS = 2
_DEFAULT_MEMORY_CPU_RATIO = 1


def instance_type_exists(instance_type: str) -> bool:
    """Check if the given instance type is valid for LSF."""
    return lsf_utils.LsfInstanceType.is_valid_instance_type(instance_type)


def get_default_instance_type(
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None) -> Optional[str]:
    # Delete unused parameters.
    del disk_tier, region, zone, local_disk, use_spot, max_hourly_cost

    # LSF provisions resources via `-n` (slots) and `rusage[mem=...]`.
    # Unlike Slurm, a memory request is always emitted: site esub scripts
    # effectively mandate rusage[mem=...] (they warn and inject a default
    # otherwise), so there is no memory=0 special case.
    instance_cpus = float(
        cpus.strip('+')) if cpus is not None else _DEFAULT_NUM_VCPUS
    if memory is not None:
        if memory.endswith('+'):
            instance_mem = float(memory[:-1])
        elif memory.endswith('x'):
            instance_mem = float(memory[:-1]) * instance_cpus
        else:
            instance_mem = float(memory)
    else:
        instance_mem = instance_cpus * _DEFAULT_MEMORY_CPU_RATIO
    virtual_instance_type = lsf_utils.LsfInstanceType(instance_cpus,
                                                      instance_mem).name
    return virtual_instance_type


def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    """List accelerators in LSF clusters.

    Returns a dictionary mapping GPU type to a list of InstanceTypeInfo
    objects.
    """
    return list_accelerators_realtime(gpus_only, name_filter, region_filter,
                                      quantity_filter, case_sensitive,
                                      all_regions, require_price)[0]


def list_accelerators_realtime(
    gpus_only: bool = True,
    name_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    quantity_filter: Optional[int] = None,
    case_sensitive: bool = True,
    all_regions: bool = False,
    require_price: bool = False,
) -> Tuple[Dict[str, List[common.InstanceTypeInfo]], Dict[str, int], Dict[str,
                                                                          int]]:
    """Fetches real-time accelerator information from the LSF cluster(s).

    GPU capacity and availability come from `bhosts -gpu -w` (one row per
    physical GPU; NJOBS==0 means free).

    Args:
        gpus_only: If True, only return GPU accelerators.
        name_filter: Regex filter for accelerator names (e.g., 'V100').
        region_filter: Optional filter for LSF clusters.
        quantity_filter: Minimum number of accelerators required per host.
        case_sensitive: Whether name_filter is case-sensitive.
        all_regions: Unused in the LSF context.
        require_price: Unused in the LSF context.

    Returns:
        A tuple of three dictionaries:
        - qtys_map: Maps GPU type to a list of InstanceTypeInfo objects for
          the counts available per host.
        - total_capacity: Maps GPU type to total count across all hosts.
        - total_available: Maps GPU type to total free count across all
          hosts.
    """
    del gpus_only, all_regions, require_price

    enabled_clouds = sky_check.get_cached_enabled_clouds_or_refresh(
        cloud.CloudCapability.COMPUTE)
    if not sky_clouds.cloud_in_iterable(sky_clouds.Lsf(), enabled_clouds):
        return {}, {}, {}

    if region_filter is None:
        clusters_to_query = lsf_utils.get_all_lsf_cluster_names()
        if not clusters_to_query:
            return {}, {}, {}
    else:
        clusters_to_query = [region_filter]

    qtys_map: Dict[str,
                   Set[common.InstanceTypeInfo]] = collections.defaultdict(set)
    total_capacity: Dict[str, int] = collections.defaultdict(int)
    total_available: Dict[str, int] = collections.defaultdict(int)

    regex_flags = 0 if case_sensitive else re.IGNORECASE

    for cluster in clusters_to_query:
        try:
            client = lsf_utils.make_client(cluster)
            gpu_rows = client.get_gpu_hosts()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f'Skipping LSF cluster {cluster!r} while '
                           f'collecting GPU info: {e}')
            continue

        # Group per (host, canonical model): count total and free GPUs.
        per_host: Dict[Tuple[str, str],
                       List[bool]] = (collections.defaultdict(list))
        for row in gpu_rows:
            gpu_type = lsf_utils.canonicalize_lsf_gpu_model(row.model)
            per_host[(row.host, gpu_type)].append(row.is_free)

        pricing = _get_pricing(region=cluster)
        for (_, gpu_type), frees in per_host.items():
            host_total = len(frees)
            host_free = sum(frees)

            if name_filter and not re.match(
                    name_filter, gpu_type, flags=regex_flags):
                continue
            if quantity_filter and host_total < quantity_filter:
                continue

            per_accel = common.get_hourly_cost_from_pricing(
                pricing,
                cpus=0,
                memory=0,
                accelerator_name=gpu_type,
                accelerator_count=1,
            )
            # Generate powers-of-2 GPU counts up to host_total, plus the
            # actual total if it is not a power of 2.
            counts = []
            count = 1
            while count <= host_total:
                counts.append(count)
                count *= 2
            if counts and counts[-1] != host_total:
                counts.append(host_total)

            for cnt in counts:
                qtys_map[gpu_type].add(
                    common.InstanceTypeInfo(
                        instance_type=None,
                        accelerator_name=gpu_type,
                        accelerator_count=cnt,
                        cpu_count=None,
                        memory=None,
                        price=per_accel * cnt,
                        region=cluster,
                        cloud='lsf',
                        device_memory=0.0,
                        spot_price=per_accel * cnt,
                    ))

            total_capacity[gpu_type] += host_total
            total_available[gpu_type] += host_free

    if not total_capacity:
        err_msg = 'No matching GPU hosts found in the LSF cluster'
        filters_applied = []
        if name_filter:
            filters_applied.append(f'gpu_name={name_filter!r}')
        if quantity_filter:
            filters_applied.append(f'quantity>={quantity_filter}')
        if filters_applied:
            err_msg += f' with filters ({", ".join(filters_applied)})'
        err_msg += '.'
        logger.error(err_msg)
        raise ValueError(err_msg)

    final_qtys_map = {
        gpu: sorted(instances, key=lambda x: x.accelerator_count)
        for gpu, instances in qtys_map.items()
    }

    logger.debug(f'Aggregated LSF GPU Info: '
                 f'qtys={final_qtys_map}, '
                 f'capacity={dict(total_capacity)}, '
                 f'available={dict(total_available)}')

    return final_qtys_map, dict(total_capacity), dict(total_available)


def _get_pricing(region: Optional[str], zone: Optional[str] = None) -> Dict:
    """Resolve the pricing dict for an LSF cluster/queue from config.

    Each level is deep-merged into the previous so that partial overrides
    inherit unset keys from the parent level:

        cloud-level  <  cluster-level  <  queue-level
    """
    paths: List[Tuple[str, ...]] = [('lsf', 'pricing')]
    if region is not None:
        paths.append(('lsf', 'cluster_configs', region, 'pricing'))
    if region is not None and zone is not None:
        paths.append(('lsf', 'cluster_configs', region, 'queue_configs', zone,
                      'pricing'))
    return common.resolve_pricing_config(*paths)


def get_hourly_cost(instance_type: str,
                    use_spot: bool,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the hourly cost for an LSF virtual instance type.

    Pricing is read from the ``lsf.pricing`` section of
    ``~/.sky/config.yaml``.
    """
    del use_spot  # LSF has no spot pricing.
    instance = lsf_utils.LsfInstanceType.from_instance_type(instance_type)
    return common.get_hourly_cost_from_pricing(
        _get_pricing(region, zone),
        cpus=instance.cpus,
        memory=instance.memory,
        accelerator_name=instance.accelerator_type,
        accelerator_count=instance.accelerator_count,
    )


def validate_region_zone(
    region_name: Optional[str],
    zone_name: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    return (region_name, zone_name)
