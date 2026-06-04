"""
================================================================================
EBSD 3D GRAIN LABELING MODULE
================================================================================

This module implements high-performance grain segmentation for 2D or 3D Electron
Backscatter Diffraction (EBSD) data using Numba JIT compilation.

ALGORITHM OVERVIEW:
  - Performs connected-component labeling on 3D voxel grids
  - Groups neighboring voxels into "grains" based on:
    * Same crystallographic phase
    * Misorientation angle below a user-defined tolerance (typically 2–15°)
  - Uses quaternion representation for crystal orientations
  - Accounts for crystal symmetry via symmetry quaternion operations
  - Supports both 2/6-NN (face-sharing) and 9/27-NN (full cube) connectivity

================================================================================
"""

import numpy as np
import numba as nb

from pyebsdindex.rotlib import quat_multiply


# ============================================================================
# PUBLIC API: High-level entry points
# ============================================================================

def grainlabel(pyindxdata, indxer, mistol=3.0,
               mask=None, labels=None, nnonly=False, dimensions=None):
  """
  High-level grain labeling interface for PyEBSDindex results.

  Extracts quaternions, phase IDs, and symmetry operators from an indexer
  object.

  Parameters
  ----------
  pyindxdata : result array from pyebsdindex.index_pats() indexing results;
      Labels will be made for each phase and the most-likely phase
      identifier, pyindxdata[-1].  Each entry in the list requires
      the fields:
      - 'quat' : ndarray of shape (npts, 4) — quaternion orientations
      - 'phase' : ndarray of shape (npts,) — phase ID per point
  indxer : object
      Indexer object with attributes:
      - phaseLib : list of phase objects, each with .qsymops (symmetry quaternions)
      - fID : file/scan metadata with .nRows, .nCols for the scan.
  mistol : float, optional
      Misorientation tolerance in degrees (default: 2.0)
  mask : ndarray, optional
      Boolean or binary mask; voxels where mask == 0 are pre-labeled as -2, and will
      not be labeled.
  labels : ndarray, optional
      Pre-existing label array; if None, initialized to -1 (unlabeled)
  nnonly : bool, optional
      If True, use 4/6-NN (face-sharing); if False, use 9/27-NN (default: False)
  dimensions : list or array, optional
      [nRows, nCols] scan dimensions; extracted from indxer if not provided

  Returns
  -------
  labels : a list of length nphases+1 of ndarray of shape (nRows, nCols) or (nSlices, nRows, nCols) for each phase
      and for the pyindxdata[-1] (use the most likely phase identifier).
      Integer grain labels; -1 = unlabeled, ≥0 = grain ID

  Notes
  -----
  - Automatically iterates over all phases in phaseLib
  - Reshapes 1D/2D data to 3D for uniform processing
  """

  qsym = []
  for p in indxer.phaseLib:
    qsym.append(p.qsymops)

  nphases = len(qsym)

  if dimensions is None:
    dimensions = [indxer.fID.nRows, indxer.fID.nCols]
  else:
    dimensions = dimensions


  labels = []
  for i in range(nphases):
    quats = pyindxdata[i]['quat']
    phaseID = (pyindxdata[i]['phase']).clip(None, 0)
    l = phaselabel(quats, phaseID, mistol, [qsym[i]], dimensions,
                    labels=None, nnonly=nnonly, mask=mask)
    labels.append(l)

  quats = pyindxdata[-1]['quat']
  phaseID = (pyindxdata[-1]['phase'])
  l = phaselabel(quats, phaseID, mistol, qsym, dimensions,
                 labels=None, nnonly=nnonly, mask=mask)
  labels.append(l)
  return labels

def appendgrainlabels(pyindxdata, labels):
  nphase = len(pyindxdata)-1

  for i in range(nphase):
    pyindxdata[i]['grainid'][:] = labels[i].flatten()

  pyindxdata[-1]['grainid'][:] = labels[-1].flatten()

  return pyindxdata


def phaselabel(quats, phaseID, mistol, quatsym, dimensions,
               labels=None, nnonly=False, mask=None):
  """
  Core grain labeling function for 2D or 3D EBSD data.

  Reshapes input data to 3D, initializes label array, and iterates over
  each crystallographic phase, calling label3D() for each.

  Parameters
  ----------
  quats : ndarray of shape (npts, 4)
      Quaternion orientations (w, x, y, z convention)
  phaseID : ndarray of shape (npts,)
      Phase ID per voxel (integer, typically 0–N_phases)
  mistol : float
      Misorientation tolerance in degrees
  quatsym : list of ndarray or single ndarray
      Symmetry quaternions per phase:
      - If list: quatsym[i] has shape (nsym_i, 4) for phase i.
      - If ndarray: single set of symmetries applied to all phases
  dimensions : list of int
      [nRows, nCols] for 2D data, or [nSlices, nRows, nCols] for 3D
  labels : ndarray, optional
      Pre-existing label array; if None, initialized to -1
  nnonly : bool, optional
      If True, use 4/6-NN; if False, use 9/27-NN (default: False)
  mask : ndarray, optional
      Binary mask; voxels where mask == 0 are labeled -1 (masked out)
      indicating no label should be applied to these points.  Useful for
      alternate non-indexed point thresholds.

  Returns
  -------
  labels : ndarray
      Grain labels, reshaped to match input dimensions
      - -1 : unlabeled (no grain assigned)
      - ≥0 : grain ID

  Notes
  -----
  - Automatically pads 2D data to 3D (nslice=1) for uniform processing
  - Iterates over phases in order; each phase gets its own label sequence
  - Modifies labels in-place for efficiency
  """
  ndim = len(dimensions)

  # ---- Pad dimensions to 3D ----
  dim = np.ones(3, dtype=np.int64)
  dim[-ndim:] = np.array(dimensions, dtype=np.int64)

  # ---- Reshape quaternions to 3D ----
  shpq = quats.shape
  q = quats.reshape(dim[0], dim[1], dim[2], 4)
  phaseID = phaseID.reshape(dim[0], dim[1], dim[2])
  unqphase = np.unique(phaseID[phaseID >= 0])
  nphase = len(unqphase)



  # ---- Initialize or reshape label array ----
  if labels is None:
    labels = -1 * np.ones((dim[0], dim[1], dim[2]), dtype=np.int64)
  else:
    labels = labels.reshape(dim[0], dim[1], dim[2])

  # ---- Apply mask (if provided) ----
  if mask is not None:
    mask = mask.reshape(dim[0], dim[1], dim[2])
    labels[mask == 0] = -2

  # --- mark the unindexed points --- #
  # --- this is likely not necessary as PhaseID <= -1 are not labeled ---#
  labels[phaseID < 0] = -2

  # ---- Normalize quatsym to list ----
  if isinstance(quatsym, np.ndarray):
    qsym = [quatsym]
  else:
    qsym = quatsym


  if len(qsym) != nphase:
    raise ValueError('qsym must be a list of length equal to the number of phases')

  # ---- Iterate over phases ----
  phase = 0
  for qsym_i in qsym:
    mxl = labels.max() + 1
    labels = label3D(q, phaseID, labels, mistol, qsym_i, phase, mxl, nnonly=nnonly)
    phase += 1

  # ---- Reshape back to original dimensions ----
  if ndim == 2:
    labels = labels.reshape(dim[1], dim[2])
  labels = labels.clip(-1)
  return labels


# ============================================================================
# HELPER: Quaternion Misorientation with Symmetry
# ============================================================================

@nb.njit(fastmath=True, inline='always', boundscheck=False, error_model='numpy')
def misquatdot(q1_0, q1_1, q1_2, q1_3,
               q2_0, q2_1, q2_2, q2_3,
               qsym, tol):
  """
  Tests if the misorientation between two quaternions is within tolerance,
  accounting for crystal symmetry.

  Computes the minimum misorientation angle between q1 and q2 by testing
  symmetry-equivalent orientations of q2. Returns 1 if any symmetry
  variant passes the tolerance check; otherwise 0. This will exit early if
  any symmetry variant passes the tolerance check.

  MATHEMATICAL BACKGROUND:
    - Quaternion dot product: dot(q1, q2) = w1*w2 + x1*x2 + y1*y2 + z1*z2
    - Misorientation angle θ relates to dot product via:
      cos(θ/2) = |dot(q1, q2)|  (absolute value accounts for q ≡ -q)
    - Tolerance is pre-converted to cos(θ_tol/2) by the caller
    - Crystal symmetry: test q1 · (q2 * sym) for symmetry ops

  Parameters
  ----------
  q1_0, q1_1, q1_2, q1_3 : float
      Components of first quaternion (w, x, y, z)
  q2_0, q2_1, q2_2, q2_3 : float
      Components of second quaternion (w, x, y, z)
  qsym : ndarray of shape (nsym, 4)
      Symmetry quaternions; qsym[0] SHOULD be identity [1, 0, 0, 0]
  tol : float
      Tolerance as cos(θ_tol/2), where θ_tol is the misorientation angle limit

  Returns
  -------
  int
      1 if misorientation is within tolerance (accounting for symmetry)
      0 otherwise

  Optimization Notes
  ------------------
  - Takes scalar components (avoids array view overhead in Numba)
  - Fast path: checks identity symmetry first (most grains match without rotation)
  - Inline compilation: function is inlined at call sites
  - fastmath=True: allows unsafe floating-point optimizations

  """

  # ---- Fast path: identity symmetry (qsym[0] = [1, 0, 0, 0]) ----
  # Most neighboring voxels inside a grain match without needing symmetry rotation.
  # This avoids the full loop for the common case.
  dot0 = abs(q1_0 * q2_0 + q1_1 * q2_1 + q1_2 * q2_2 + q1_3 * q2_3)
  #print(dot0)
  if dot0 >= tol:
    return 1

  # ---- Full symmetry loop ----
  nsym = qsym.shape[0]
  # simple iteration in case identity is NOT first entry -- for PyEBSDindex it always is.
  # This should exit early, so should not cause a significant delay.
  startstop = [[1, nsym], [0,1]]
  for itrange in startstop:
    for i in range(itrange[0], itrange[1]):
      s0 = qsym[i, 0]
      s1 = qsym[i, 1]
      s2 = qsym[i, 2]
      s3 = qsym[i, 3]

      # --- w,x,y,z = quat_multiply(qsym_i, q2) ---
      w = s0 * q2_0 - (s1 * q2_1 + s2 * q2_2 + s3 * q2_3)
      x = s0 * q2_1 + q2_0 * s1 + (s2 * q2_3 - s3 * q2_2)
      y = s0 * q2_2 + q2_0 * s2 + (s3 * q2_1 - s1 * q2_3)
      z = s0 * q2_3 + q2_0 * s3 + (s1 * q2_2 - s2 * q2_1)

      # Dot product with q1
      dot = abs(q1_0 * w + q1_1 * x + q1_2 * y + q1_3 * z)
      if dot >= tol:
        return 1

  return 0


# ============================================================================
# MAIN ALGORITHM: 3D EBSD Grain Labeling
# ============================================================================

@nb.njit(parallel=False, fastmath=True, boundscheck=False, error_model='numpy')
def label3D(q, phaseID, labels, mistol, qsym, phase, initialID, nnonly=False):
  """
  Numba-optimized 3D connected-component grain labeling for EBSD data.

  Implements a depth-first search (DFS) flood-fill algorithm using a
  dynamically-growing 1D stack. For each unlabeled voxel of the target phase,
  seeds a new grain and grows it by checking neighbors for:
    1. Same phase ID
    2. Misorientation angle below tolerance (accounting for symmetry)

  ALGORITHM PSEUDOCODE:
  ┌──────────────────────────────────────────────────────────────────┐
  │ for each voxel (z, y, x) in 3D grid:                             │
  │   if phaseID[z,y,x] == phase AND labels[z,y,x] == -1:            │
  │     labels[z,y,x] ← labeli                                       │
  │     push (z,y,x) onto stack                                      │
  │     while stack not empty:                                       │
  │       pop (cz, cy, cx)                                           │
  │       for each neighbor (nz, ny, nx):                            │
  │         if labels[nz,ny,nx] == -1 AND phaseID[nz,ny,nx] == phase:│
  │           if misorientation(q[cz,cy,cx], q[nz,ny,nx]) < tol:     │
  │             labels[nz,ny,nx] ← labeli                            │
  │             push (nz, ny, nx) onto stack                         │
  │     labeli ← labeli + 1                                          │
  └──────────────────────────────────────────────────────────────────┘

  Parameters
  ----------
  q : ndarray of shape (nslice, nrow, ncol, 4), dtype float32
      Quaternion orientations (w, x, y, z) at each voxel
  phaseID : ndarray of shape (nslice, nrow, ncol), dtype int32
      Crystallographic phase ID at each voxel
  labels : ndarray of shape (nslice, nrow, ncol), dtype int64
      Output grain labels (modified in-place)
      - -1 : unlabeled
      - -2 : masked out (pre-set by caller)
      - ≥0 : grain ID
  mistol : float
      Misorientation tolerance in degrees (e.g., 2.0, 5.0, 15.0)
  qsym : ndarray of shape (nsym, 4), dtype float32
      Symmetry quaternions for the current phase
      CRITICAL: qsym[0] MUST be the identity quaternion [1, 0, 0, 0]
  phase : int32
      Phase ID to label in this call
  initialID : uint64
      Starting label ID for this phase (typically max(labels) + 1)
  nnonly : bool, optional
      If True, use 6-NN (face-sharing neighbors only)
      If False, use 27-NN (full 3×3×3 cube, default)

  Returns
  -------
  labels : ndarray
      Updated label array (same object, modified in-place)

  CONNECTIVITY MODES
  ------------------
  4/6-NN (nnonly=True):
    Neighbors: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1) — face-sharing only
    Use case: Stricter grain boundaries, fewer spurious merges

  9/27-NN (nnonly=False, default):
    Neighbors: all (dz, dy, dx) ∈ {-1, 0, 1}³ except (0, 0, 0)
    Use case: Standard EBSD analysis, captures edge/corner contacts
  """

  nslice, nrow, ncol = labels.shape
  nyx = nrow * ncol  # Precomputed stride for flattening (z-major order)

  # ---- Convert tolerance from degrees to cosine space ----
  # θ_tol (degrees) → cos(θ_tol/2) for quaternion dot product comparison
  tol = np.cos(mistol * np.pi / 180.0 / 2.0)
  labeli = initialID

  # ---- Dynamic 1D stack (flattened indices) ----
  # Initialized with capacity = 2*ncol; grows as needed
  INITIAL_CAPACITY = 2*ncol
  pts2check = np.empty(INITIAL_CAPACITY, dtype=np.int64)

  # ---- Main loop: iterate over all voxels ----
  for z in range(nslice):
    for y in range(nrow):
      for x in range(ncol):

        # Skip if not target phase
        if phaseID[z, y, x] != phase:
          continue
        # Skip if already labeled (or masked out)
        if labels[z, y, x] != -1:
          continue

        # ---- Seed new grain ----
        labels[z, y, x] = labeli
        pts2check[0] = z * nyx + y * ncol + x
        stack_size = 1

        # ---- Flood fill (DFS) ----
        while stack_size > 0:
          stack_size -= 1
          flat = pts2check[stack_size]

          # Decompose flat index back to (z, y, x)
          cz = flat // nyx
          rem = flat - cz * nyx
          cy = rem // ncol
          cx = rem - cy * ncol

          # Hoist current quaternion (loaded ONCE per pop)
          # Avoids repeated array indexing in the neighbor loop
          qc0 = q[cz, cy, cx, 0]
          qc1 = q[cz, cy, cx, 1]
          qc2 = q[cz, cy, cx, 2]
          qc3 = q[cz, cy, cx, 3]

          if not nnonly:
            # =============================
            # 27-Nearest Neighbors (3×3×3 cube)
            # =============================
            for dz in range(-1, 2):
              nz = cz + dz
              if nz < 0 or nz >= nslice:
                continue
              for dy in range(-1, 2):
                ny = cy + dy
                if ny < 0 or ny >= nrow:
                  continue
                for dx in range(-1, 2):
                  nx = cx + dx
                  if nx < 0 or nx >= ncol:
                    continue
                  # Skip if already labeled
                  if labels[nz, ny, nx] != -1:
                    continue
                  # Skip if different phase
                  if phaseID[nz, ny, nx] != phase:
                    continue

                  # Test misorientation
                  if misquatdot(
                          q[nz, ny, nx, 0], q[nz, ny, nx, 1],
                          q[nz, ny, nx, 2], q[nz, ny, nx, 3],
                          qc0, qc1, qc2, qc3,
                          qsym, tol
                  ) > 0:
                    # Assign label and push to stack
                    labels[nz, ny, nx] = labeli
                    if stack_size >= pts2check.shape[0]:
                      pts2check = _grow_stack_1d(pts2check)
                    pts2check[stack_size] = nz * nyx + ny * ncol + nx
                    stack_size += 1
          else:
            # =============================
            # 6-Nearest Neighbors (face-sharing only)
            # =============================
            # Unrolled for performance: avoids loop overhead

            # -Z neighbor
            nz = cz - 1
            if nz >= 0 and labels[nz, cy, cx] == -1 and phaseID[nz, cy, cx] == phase:
              if misquatdot(
                      q[nz, cy, cx, 0], q[nz, cy, cx, 1],
                      q[nz, cy, cx, 2], q[nz, cy, cx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[nz, cy, cx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = nz * nyx + cy * ncol + cx
                stack_size += 1

            # +Z neighbor
            nz = cz + 1
            if nz < nslice and labels[nz, cy, cx] == -1 and phaseID[nz, cy, cx] == phase:
              if misquatdot(
                      q[nz, cy, cx, 0], q[nz, cy, cx, 1],
                      q[nz, cy, cx, 2], q[nz, cy, cx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[nz, cy, cx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = nz * nyx + cy * ncol + cx
                stack_size += 1

            # -Y neighbor
            ny = cy - 1
            if ny >= 0 and labels[cz, ny, cx] == -1 and phaseID[cz, ny, cx] == phase:
              if misquatdot(
                      q[cz, ny, cx, 0], q[cz, ny, cx, 1],
                      q[cz, ny, cx, 2], q[cz, ny, cx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[cz, ny, cx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = cz * nyx + ny * ncol + cx
                stack_size += 1

            # +Y neighbor
            ny = cy + 1
            if ny < nrow and labels[cz, ny, cx] == -1 and phaseID[cz, ny, cx] == phase:
              if misquatdot(
                      q[cz, ny, cx, 0], q[cz, ny, cx, 1],
                      q[cz, ny, cx, 2], q[cz, ny, cx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[cz, ny, cx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = cz * nyx + ny * ncol + cx
                stack_size += 1

            # -X neighbor
            nx = cx - 1
            if nx >= 0 and labels[cz, cy, nx] == -1 and phaseID[cz, cy, nx] == phase:
              if misquatdot(
                      q[cz, cy, nx, 0], q[cz, cy, nx, 1],
                      q[cz, cy, nx, 2], q[cz, cy, nx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[cz, cy, nx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = cz * nyx + cy * ncol + nx
                stack_size += 1

            # +X neighbor
            nx = cx + 1
            if nx < ncol and labels[cz, cy, nx] == -1 and phaseID[cz, cy, nx] == phase:
              if misquatdot(
                      q[cz, cy, nx, 0], q[cz, cy, nx, 1],
                      q[cz, cy, nx, 2], q[cz, cy, nx, 3],
                      qc0, qc1, qc2, qc3, qsym, tol) > 0:
                labels[cz, cy, nx] = labeli
                if stack_size >= pts2check.shape[0]:
                  pts2check = _grow_stack_1d(pts2check)
                pts2check[stack_size] = cz * nyx + cy * ncol + nx
                stack_size += 1

        # ---- Increment label for next grain ----
        labeli += 1

  return labels


# ============================================================================
# UTILITY: Dynamic Stack Management
# ============================================================================

@nb.njit(inline='always', boundscheck=False)
def _grow_stack_1d(stack):
  """
  Double the capacity of a 1D int64 stack buffer and copy contents.

  Called when the stack is full and a new element needs to be pushed.
  Allocates a new array with 2× capacity and copies all existing elements.

  Parameters
  ----------
  stack : ndarray of shape (n,), dtype int64
      Current stack buffer

  Returns
  -------
  new_stack : ndarray of shape (2*n,), dtype int64
      Enlarged stack with contents copied

  """
  old_cap = stack.shape[0]
  new_stack = np.empty(old_cap * 2, dtype=np.int64)
  for i in range(old_cap):
    new_stack[i] = stack[i]
  return new_stack