/**
 * Infrastructure utility functions for context key generation
 */

/**
 * Builds a context stats key for use in contextStats mapping.
 * This key is used consistently across the application to identify
 * infrastructure contexts (Kubernetes, SSH Node Pools, Slurm clusters).
 *
 * @param {string} contextName - The context name (e.g., 'my-context', 'ssh-pool1', 'slurm-cluster')
 * @param {Object} options - Options for determining the context type
 * @param {boolean} [options.isSSH] - Whether this is an SSH Node Pool context
 * @param {boolean} [options.isSlurm] - Whether this is a Slurm cluster context
 * @param {string} [options.scheduler] - Batch scheduler kind ('slurm' | 'lsf')
 * @param {string} [options.cloud] - Cloud type ('Kubernetes', 'SSH', 'slurm', 'Slurm', or 'LSF')
 * @returns {string} - The context stats key (e.g., 'kubernetes/my-context', 'ssh/pool1', 'slurm/cluster', 'lsf/cluster')
 */
export function buildContextStatsKey(contextName, options = {}) {
  if (!contextName) {
    return null;
  }

  const { isSSH, isSlurm, scheduler, cloud } = options;
  const cloudLower = cloud?.toLowerCase();

  // Determine context type from options or infer from context name
  let contextType = null;
  if (isSSH || cloud === 'SSH') {
    contextType = 'ssh';
  } else if (isSlurm || scheduler === 'slurm' || cloudLower === 'slurm') {
    contextType = 'slurm';
  } else if (scheduler === 'lsf' || cloudLower === 'lsf') {
    // Without this an LSF cluster keys as 'kubernetes/<name>', which both
    // collides with a Kubernetes context of the same name and files the
    // cluster's counts under the wrong section.
    contextType = 'lsf';
  } else if (cloud === 'Kubernetes') {
    contextType = 'kubernetes';
  } else if (contextName.startsWith('ssh-')) {
    // Infer from context name if no explicit type provided
    contextType = 'ssh';
  } else {
    // Default to Kubernetes for backward compatibility
    contextType = 'kubernetes';
  }

  // Process context name based on type
  let processedName = contextName;
  if (contextType === 'ssh') {
    // Remove 'ssh-' prefix if present
    processedName = contextName.replace(/^ssh-/, '');
  }

  return `${contextType}/${processedName}`;
}

/**
 * Builds a context stats key from a cloud/region pair (used in jobs and clusters).
 * This is a convenience wrapper around buildContextStatsKey for the common
 * pattern where we have a cloud type and region/context name.
 *
 * @param {string} cloud - Cloud type ('Kubernetes', 'SSH', 'slurm', 'Slurm', or 'LSF')
 * @param {string} region - Region/context name (may include 'ssh-' prefix for SSH)
 * @returns {string|null} - The context stats key or null if invalid
 */
export function buildContextStatsKeyFromCloud(cloud, region) {
  if (!cloud || !region) {
    return null;
  }
  return buildContextStatsKey(region, { cloud });
}
