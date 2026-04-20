#pragma once
// ---------------------------------------------------------------------------
// cooperative_barrier.cuh — Grid-level synchronization primitives.
//
// Wraps cooperative_groups::grid_group::sync() for use inside megakernel.cu.
// All syncs are unconditional — all thread blocks must reach every barrier.
// Conditional grid.sync() causes permanent deadlock.
//
// FlashFormer alternative (arXiv 2505.22758): per-thread fence + atomic.
// Not used here; grid.sync() is simpler and available on CC >= 6.0.
//
// Safety notes:
//   - Never call GRID_SYNC inside divergent branches.
//   - GRID_SYNC is only valid inside __global__ functions launched via
//     cudaLaunchCooperativeKernel. Calling from regular kernels = deadlock.
// ---------------------------------------------------------------------------

#pragma once
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// GRID_SYNC: barrier across all thread blocks in the cooperative launch.
// Must be called by ALL warps in ALL blocks unconditionally.
#define GRID_SYNC(grid) (grid).sync()

// BLOCK_SYNC: __syncthreads() alias for clarity.
#define BLOCK_SYNC() __syncthreads()
