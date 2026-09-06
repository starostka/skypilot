// The dashboard cache is a pass-through here so the connector's own logic is
// what's under test, not the caching layer.
jest.mock('@/lib/cache', () => ({
  __esModule: true,
  default: { get: jest.fn((fn, args = []) => fn(...args)) },
}));

jest.mock('@/data/connectors/client', () => ({
  __esModule: true,
  apiClient: { post: jest.fn(), get: jest.fn() },
}));

import { apiClient } from '@/data/connectors/client';
import {
  getLsfInfrastructure,
  getSlurmInfrastructure,
} from '@/data/connectors/infra';

// Every Slurm endpoint answers the same way: POST returns a request id in a
// header, then /api/get returns the result under a JSON-encoded return_value.
const scheduled = (requestId) => ({
  ok: true,
  status: 200,
  headers: { get: () => requestId },
});

const result = (returnValue) => ({
  ok: true,
  status: 200,
  json: async () => ({ return_value: JSON.stringify(returnValue) }),
});

// The Infra page must list a cluster that is configured but currently
// unreachable, so the configured-cluster query has to be independent of the
// node and GPU queries rather than derived from them.
describe('getSlurmInfrastructure configured clusters', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.post.mockImplementation(async (path) => {
      if (path === '/slurm_cluster_names') return scheduled('req-clusters');
      if (path === '/slurm_gpu_availability') return scheduled('req-gpus');
      if (path === '/slurm_node_info') return scheduled('req-nodes');
      throw new Error(`unexpected POST to ${path}`);
    });
  });

  it('reports a configured cluster whose node and GPU queries are empty', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return result(['offline-cluster']);
      return result([]);
    });

    const data = await getSlurmInfrastructure();

    expect(data.slurmClusterNames).toEqual(['offline-cluster']);
    expect(data.perNodeSlurmGPUs).toEqual([]);
    expect(data.perClusterSlurmGPUs).toEqual([]);
  });

  it('does not hold the node and GPU queries behind the cluster list', async () => {
    // Assertions stay in the test body: the connector catches everything a
    // mock throws, so an assertion made inside one would be swallowed and the
    // test would pass regardless.
    let releaseClusters;
    const clusterListPending = new Promise((resolve) => {
      releaseClusters = resolve;
    });
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) {
        await clusterListPending;
        return result(['cluster-a']);
      }
      return result([]);
    });

    const pending = getSlurmInfrastructure();
    await new Promise((resolve) => setTimeout(resolve, 0));

    // The cluster list is still in flight, yet the other two have already got
    // past their own POST to fetching a result. Serializing them behind it
    // would leave these uncalled.
    const fetched = apiClient.get.mock.calls.map(([path]) => path);
    expect(fetched).toEqual(
      expect.arrayContaining([
        expect.stringContaining('req-nodes'),
        expect.stringContaining('req-gpus'),
      ])
    );

    releaseClusters();
    expect((await pending).slurmClusterNames).toEqual(['cluster-a']);
  });

  it('falls back to an empty list when the cluster query fails', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return { ok: false, status: 500 };
      return result([]);
    });

    const data = await getSlurmInfrastructure();

    expect(data.slurmClusterNames).toEqual([]);
  });
});

// LSF answers the same request-id protocol, with a fourth endpoint: queues,
// which the server reads from config rather than from the hosts.
describe('getLsfInfrastructure', () => {
  const queueRows = [
    {
      lsf_cluster_name: 'dtu',
      queue: 'hpc',
      is_default: true,
      gpu_type: null,
      gpu_count_per_host: null,
      status: 'Open:Active',
      pend: 3,
      run: 12,
    },
    {
      lsf_cluster_name: 'dtu',
      queue: 'gpuv100',
      is_default: false,
      gpu_type: 'V100',
      gpu_count_per_host: 4,
      status: 'Open:Active',
      pend: 0,
      run: 2,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.post.mockImplementation(async (path) => {
      if (path === '/lsf_cluster_names') return scheduled('req-clusters');
      if (path === '/lsf_gpu_availability') return scheduled('req-gpus');
      if (path === '/lsf_node_info') return scheduled('req-nodes');
      if (path === '/lsf_queue_info') return scheduled('req-queues');
      throw new Error(`unexpected POST to ${path}`);
    });
  });

  it('groups queues by cluster and keys nodes by lsf_cluster_name', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return result(['dtu']);
      if (path.includes('req-queues')) return result(queueRows);
      if (path.includes('req-nodes')) {
        return result([
          {
            node_name: 'n-1',
            lsf_cluster_name: 'dtu',
            queue: '',
            node_state: 'ok',
            gpu_type: 'V100',
            total_gpus: 4,
            free_gpus: 1,
          },
        ]);
      }
      return result([['dtu', [['V100', [1, 2, 4], 8, 3]]]]);
    });

    const data = await getLsfInfrastructure();

    expect(data.lsfClusterNames).toEqual(['dtu']);
    expect(data.lsfQueues.dtu.map((q) => q.name)).toEqual(['hpc', 'gpuv100']);
    expect(data.lsfQueues.dtu[0].isDefault).toBe(true);
    expect(data.lsfQueues.dtu[1].gpu_count_per_host).toBe(4);
    expect(data.perNodeLsfGPUs[0]).toMatchObject({
      node_name: 'n-1',
      cluster: 'dtu',
      gpu_name: 'V100',
      gpu_free: 1,
    });
    expect(data.allLsfGPUs).toEqual([
      { gpu_name: 'V100', gpu_total: 8, gpu_free: 3 },
    ]);
  });

  it('lists a configured cluster whose login node answers nothing', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return result(['dtu']);
      // Queue, node and GPU queries all need the login node.
      return { ok: false, status: 500 };
    });

    const data = await getLsfInfrastructure();

    expect(data.lsfClusterNames).toEqual(['dtu']);
    expect(data.perNodeLsfGPUs).toEqual([]);
    expect(data.lsfQueues).toEqual({});
  });

  it('survives one dead endpoint without blanking the rest', async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path.includes('req-clusters')) return result(['dtu']);
      if (path.includes('req-queues')) return result(queueRows);
      throw new Error('login node unreachable');
    });

    const data = await getLsfInfrastructure();

    expect(data.lsfQueues.dtu).toHaveLength(2);
    expect(data.perNodeLsfGPUs).toEqual([]);
    expect(data.allLsfGPUs).toEqual([]);
  });
});
