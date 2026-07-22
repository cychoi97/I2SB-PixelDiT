import os
import glob
import numpy as np
import pydicom

from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T


EXTENSION = ['.dcm']

# ---------------------------------------------------------------------------
# Kernel alias map — Siemens SOMATOM Definition AS+ (B-series) and FORCE (Br-series)
# produce clinically equivalent images at the same sharpness level.
#   Standard / soft  : B30f  ≡  Br36d
#   Medium           : B45f  ≡  Br49d
#   Sharp            : B60f  ≡  Br59d
# ---------------------------------------------------------------------------
KERNEL_ALIAS_MAP = {
    'B30f':  ['Br36d'],
    'B45f':  ['Br49d'],
    'B60f':  ['Br59d'],
}

# ===========================================================================
# Per-domain attribute registries.
#
# Each domain has its own independent attribute index space.
# These IDs are fed to the per-domain embedding tables in
# MultiDomainEmbedder.
# ===========================================================================
DOSE_ATTRS   = {"Quarter": 0, "Full": 1}
KERNEL_ATTRS = {"B30f": 0, "B45f": 1, "B60f": 2}

DOSE_ID_TO_NAME   = {v: k for k, v in DOSE_ATTRS.items()}
KERNEL_ID_TO_NAME = {v: k for k, v in KERNEL_ATTRS.items()}

NUM_DOSE_ATTRS   = len(DOSE_ATTRS)
NUM_KERNEL_ATTRS = len(KERNEL_ATTRS)

# ===========================================================================
# DEFAULT_PAIRS — (src_dose, src_kernel, trg_dose, trg_kernel)
#
# Each entry defines a conversion direction where exactly ONE domain changes
# and the other stays fixed.  This is the default set of pairs used when
# --pair-defs is not specified on the command line.
#
# Unified directory layout (for ALL tasks):
#   <dataset_dir>/<group>/<dose>/<patient>/<kernel>/*.dcm
# ===========================================================================
DEFAULT_PAIRS = [
    # --- Kernel conversion (dose fixed, kernel changes) ---
    ("Quarter", "B45f", "Quarter", "B30f"),
    ("Quarter", "B60f", "Quarter", "B30f"),
    ("Quarter", "B60f", "Quarter", "B45f"),
    ("Full", "B45f", "Full", "B30f"),
    ("Full", "B60f", "Full", "B30f"),
    ("Full", "B60f", "Full", "B45f"),
    # --- Denoising (kernel fixed, dose changes) ---
    ("Quarter", "B30f", "Full", "B30f"),
    ("Quarter", "B45f", "Full", "B45f"),
    ("Quarter", "B60f", "Full", "B60f"),
]


def resolve_pair_defs(pair_defs_str=None):
    """Resolve which (src_dose, src_kernel, trg_dose, trg_kernel) tuples to use.

    Args:
        pair_defs_str : list of "SrcDose,SrcKernel:TrgDose,TrgKernel" strings.
                        If None, all DEFAULT_PAIRS are used.

    Returns:
        List of (src_dose, src_kernel, trg_dose, trg_kernel) tuples.
    """
    if not pair_defs_str:
        return list(DEFAULT_PAIRS)

    result = []
    for pdef in pair_defs_str:
        src_part, trg_part = pdef.split(':')
        src_dose, src_kernel = src_part.split(',')
        trg_dose, trg_kernel = trg_part.split(',')

        # Validate attributes.
        assert src_dose in DOSE_ATTRS, f"Unknown dose attr '{src_dose}'. Valid: {list(DOSE_ATTRS)}"
        assert trg_dose in DOSE_ATTRS, f"Unknown dose attr '{trg_dose}'. Valid: {list(DOSE_ATTRS)}"
        assert src_kernel in KERNEL_ATTRS, f"Unknown kernel attr '{src_kernel}'. Valid: {list(KERNEL_ATTRS)}"
        assert trg_kernel in KERNEL_ATTRS, f"Unknown kernel attr '{trg_kernel}'. Valid: {list(KERNEL_ATTRS)}"

        result.append((src_dose, src_kernel, trg_dose, trg_kernel))
    return result


def pair_label(src_dose, src_kernel, trg_dose, trg_kernel):
    """Human-readable label for a conversion pair."""
    return f"({src_dose},{src_kernel})→({trg_dose},{trg_kernel})"


class CTDataset(Dataset):
    """Multi-domain CT restoration dataset with full domain-awareness.

    Every sample carries the complete (dose, kernel) attribute tuple for
    both source and target, enabling the network to condition on ALL
    domains regardless of which domain is being converted.

    Directory layout (unified for all tasks):
        <dataset_dir>/<group>/<dose>/<patient>/<kernel>/*.dcm

    Returns:
        (clean_img, corrupt_img, src_dose_id, src_kernel_id,
         trg_dose_id, trg_kernel_id)

    Args:
        opt        : namespace with dataset_dir, image_size, pair_defs.
        log        : Logger instance.
        mode       : 'train', 'valid', or 'test'.
    """

    def __init__(self, opt, log, mode: str):
        super().__init__()
        self.mode = mode
        self.dataset_dir = Path(opt.dataset_dir) / mode
        self.image_size = opt.image_size

        # Resolve conversion pairs.
        pair_defs_str = getattr(opt, 'pair_defs', None)
        self.active_pairs = resolve_pair_defs(pair_defs_str)

        # Collect (src_path, trg_path, src_dose_id, src_kernel_id,
        #          trg_dose_id, trg_kernel_id) tuples.
        self.pairs = self._collect_all_pairs(
            str(self.dataset_dir), self.active_pairs
        )

        if len(self.pairs) == 0:
            raise IOError(
                f'No paired DICOM files found in {self.dataset_dir} '
                f'for pairs={[pair_label(*p) for p in self.active_pairs]}.'
            )

        # Log summary.
        pair_counts = {}
        for _, _, sd, sk, td, tk in self.pairs:
            key = (DOSE_ID_TO_NAME[sd], KERNEL_ID_TO_NAME[sk],
                   DOSE_ID_TO_NAME[td], KERNEL_ID_TO_NAME[tk])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        for (sd, sk, td, tk), cnt in sorted(pair_counts.items()):
            log.info(
                f"[Dataset] mode={mode}, "
                f"pair=({sd},{sk})→({td},{tk}), "
                f"files={cnt:,}"
            )
        log.info(
            f"[Dataset] mode={mode}, total={len(self.pairs):,}, "
            f"active_pairs={[pair_label(*p) for p in self.active_pairs]}"
        )

        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)  # [0, 1] → [-1, 1]
        ])

    # ------------------------------------------------------------------
    # Pair collection — unified directory scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_all_pairs(dataset_dir: str, active_pairs: list) -> list:
        """Collect all file pairs across all active conversion definitions.

        Returns:
            Sorted list of (src_path, trg_path,
                            src_dose_id, src_kernel_id,
                            trg_dose_id, trg_kernel_id) tuples.
        """
        all_pairs = []
        for src_dose, src_kernel, trg_dose, trg_kernel in active_pairs:
            pairs = CTDataset._collect_conversion_pairs(
                dataset_dir,
                src_dose, src_kernel,
                trg_dose, trg_kernel,
            )
            all_pairs.extend(pairs)
        return sorted(all_pairs, key=lambda x: (x[2], x[3], x[4], x[5], x[0]))

    @staticmethod
    def _collect_conversion_pairs(
        dataset_dir: str,
        src_dose: str,
        src_kernel: str,
        trg_dose: str,
        trg_kernel: str,
    ) -> list:
        """Collect file pairs for a single conversion direction.

        Unified directory layout:
            <dataset_dir>/<group>/<dose>/<patient>/<kernel>/*.dcm

        Handles kernel aliases: for each (src_kernel, trg_kernel) pair, we
        also search for their Br-series aliases.

        Args:
            dataset_dir : absolute path to the split dir (e.g. .../train).
            src_dose    : source dose attribute name (e.g. 'Quarter').
            src_kernel  : source kernel attribute name (e.g. 'B45f').
            trg_dose    : target dose attribute name (e.g. 'Quarter').
            trg_kernel  : target kernel attribute name (e.g. 'B30f').

        Returns:
            List of (src_path, trg_path,
                     src_dose_id, src_kernel_id,
                     trg_dose_id, trg_kernel_id) tuples.
        """
        pairs = []

        src_dose_id   = DOSE_ATTRS[src_dose]
        src_kernel_id = KERNEL_ATTRS[src_kernel]
        trg_dose_id   = DOSE_ATTRS[trg_dose]
        trg_kernel_id = KERNEL_ATTRS[trg_kernel]

        # All on-disk folder names for each kernel (primary + aliases).
        src_kernel_folders = [src_kernel] + KERNEL_ALIAS_MAP.get(src_kernel, [])
        trg_kernel_folders = [trg_kernel] + KERNEL_ALIAS_MAP.get(trg_kernel, [])

        for sk_folder in src_kernel_folders:
            # Glob: */<src_dose>/*/<src_kernel_folder>
            pattern = os.path.join(dataset_dir, '*', src_dose, '*', sk_folder)
            src_dirs = sorted(glob.glob(pattern))

            for src_dir in src_dirs:
                # Path structure: .../group/dose/patient/kernel
                # parent_of_kernel = .../group/dose/patient
                parent = os.path.dirname(src_dir)

                # Try each target kernel folder name (primary + aliases).
                for tk_folder in trg_kernel_folders:
                    if src_dose == trg_dose:
                        # Same dose → kernel conversion.
                        # Target is a sibling kernel dir under the same patient.
                        trg_dir = os.path.join(parent, tk_folder)
                    else:
                        # Different dose → denoising (or cross-domain).
                        # Replace dose component in the path.
                        # parent = .../group/src_dose/patient
                        trg_parent = parent.replace(
                            os.sep + src_dose + os.sep,
                            os.sep + trg_dose + os.sep,
                        )
                        if trg_parent == parent:
                            # Fallback for forward-slash paths.
                            trg_parent = parent.replace(
                                '/' + src_dose + '/',
                                '/' + trg_dose + '/',
                            )
                        trg_dir = os.path.join(trg_parent, tk_folder)

                    if not os.path.isdir(trg_dir):
                        continue

                    src_files = _sorted_dcm_files(src_dir)
                    trg_files = _sorted_dcm_files(trg_dir)

                    n = min(len(src_files), len(trg_files))
                    pairs.extend(
                        (s, t, src_dose_id, src_kernel_id,
                         trg_dose_id, trg_kernel_id)
                        for s, t in zip(src_files[:n], trg_files[:n])
                    )
                    break  # Found a matching target folder; skip other aliases.

        return pairs

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _file_ext(fname: str) -> str:
        return os.path.splitext(fname)[1].lower()

    @staticmethod
    def _clip_and_normalize(img: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        img = np.clip(img, min_val, max_val)
        img = (img - min_val) / (max_val - min_val)
        return img

    @staticmethod
    def _CT_preprocess(
        dcm,
        img: np.ndarray,
        window_width=None,
        window_level=None,
    ) -> np.ndarray:
        """Convert raw pixel data → normalised float in [0, 1].

        Applies Rescale Slope/Intercept to get Hounsfield Units, then
        clips and normalises.  When no windowing parameters are given,
        the full 12-bit CT HU range [-1024, 3071] is used.
        """
        intercept = dcm.RescaleIntercept
        slope     = dcm.RescaleSlope
        img = img * slope + intercept  # → HU

        if window_width is not None and window_level is not None:
            min_val = window_level - window_width / 2.0
            max_val = window_level + window_width / 2.0
        else:  # full 12-bit HU range
            min_val = -1024.0
            max_val =  3071.0

        return CTDataset._clip_and_normalize(img, min_val, max_val)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        """Return (clean_img, corrupt_img, src_dose_id, src_kernel_id,
                   trg_dose_id, trg_kernel_id).

        clean_img     : target (high quality) image tensor in [-1, 1], [1,H,W].
        corrupt_img   : source (degraded)     image tensor in [-1, 1], [1,H,W].
        src_dose_id   : int scalar tensor — dose domain index of source.
        src_kernel_id : int scalar tensor — kernel domain index of source.
        trg_dose_id   : int scalar tensor — dose domain index of target.
        trg_kernel_id : int scalar tensor — kernel domain index of target.
        """
        src_fname, trg_fname, sd, sk, td, tk = self.pairs[index]

        src_img = self._load_ct_slice(src_fname)
        trg_img = self._load_ct_slice(trg_fname)

        corrupt_img   = self.transform(src_img[:, :, np.newaxis])
        clean_img     = self.transform(trg_img[:, :, np.newaxis])
        src_dose_t    = torch.tensor(sd, dtype=torch.long)
        src_kernel_t  = torch.tensor(sk, dtype=torch.long)
        trg_dose_t    = torch.tensor(td, dtype=torch.long)
        trg_kernel_t  = torch.tensor(tk, dtype=torch.long)

        return clean_img, corrupt_img, src_dose_t, src_kernel_t, trg_dose_t, trg_kernel_t

    def _load_ct_slice(self, fname: str) -> np.ndarray:
        """Load a single DICOM (or image) file and return a float32 array."""
        ext = self._file_ext(fname)
        with open(fname, 'rb') as f:
            if ext == '.dcm':
                dcm = pydicom.dcmread(f, force=True)
                img = dcm.pixel_array.astype(np.float32)
                img = self._CT_preprocess(dcm, img)
            else:
                img = np.array(Image.open(f)).astype(np.float32)
        return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_dcm_files(directory: str) -> list:
    """Return sorted list of .dcm file paths in *directory*."""
    return sorted(
        f for f in glob.glob(os.path.join(directory, '*.dcm'))
        if os.path.splitext(f)[1].lower() == '.dcm'
    )
