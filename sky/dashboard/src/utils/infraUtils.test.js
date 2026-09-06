import {
  buildContextStatsKey,
  buildContextStatsKeyFromCloud,
} from '@/utils/infraUtils';

// Cluster and job counts are looked up by this key from two directions: the
// infra row (which knows its section) and the clusters/jobs lists (which know
// only a cloud name). Both must agree, or a section's counts silently read
// zero.
describe('buildContextStatsKey', () => {
  it('keys each infrastructure family under its own prefix', () => {
    expect(buildContextStatsKey('my-context')).toBe('kubernetes/my-context');
    expect(buildContextStatsKey('ssh-pool1', { isSSH: true })).toBe(
      'ssh/pool1'
    );
    expect(buildContextStatsKey('nebius', { scheduler: 'slurm' })).toBe(
      'slurm/nebius'
    );
    expect(buildContextStatsKey('dtu', { scheduler: 'lsf' })).toBe('lsf/dtu');
  });

  it('agrees with the cloud-side lookup for LSF', () => {
    // Without the LSF branch this fell through to 'kubernetes/dtu', which both
    // collided with a Kubernetes context of the same name and filed the
    // cluster under the wrong section.
    expect(buildContextStatsKeyFromCloud('LSF', 'dtu')).toBe(
      buildContextStatsKey('dtu', { scheduler: 'lsf' })
    );
    expect(buildContextStatsKeyFromCloud('Slurm', 'nebius')).toBe(
      buildContextStatsKey('nebius', { scheduler: 'slurm' })
    );
  });

  it('keeps accepting the isSlurm alias', () => {
    expect(buildContextStatsKey('nebius', { isSlurm: true })).toBe(
      'slurm/nebius'
    );
  });
});
