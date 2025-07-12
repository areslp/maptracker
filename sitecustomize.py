import sys, types
import importlib
import warnings

import numpy as _np
_deprecated_aliases = {
    'bool': bool,
    'int': int,
    'float': float,
    'complex': complex,
    'object': object,
}
for _name, _alias in _deprecated_aliases.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _alias)

# --- module registration ----------------------------------------------------
import mmcv as mmcv_mod
# load / dump wrappers
from mmengine.fileio import load as _file_load, dump as _file_dump
mmcv_mod.load = _file_load  # type: ignore
mmcv_mod.dump = _file_dump  # type: ignore
# ProgressBar & track_progress
from mmengine.utils import ProgressBar as _PBCls, track_iter_progress as _track_iter_progress, mkdir_or_exist as _mmengine_mkdir_or_exist
mmcv_mod.ProgressBar = _PBCls  # type: ignore
mmcv_mod.track_iter_progress = _track_iter_progress  # type: ignore
mmcv_mod.mkdir_or_exist = _mmengine_mkdir_or_exist  # type: ignore

mmcv_parallel_mod = types.ModuleType('mmcv.parallel')
sys.modules['mmcv.parallel'] = mmcv_parallel_mod
mmcv_runner_mod = types.ModuleType('mmcv.runner')
sys.modules['mmcv.runner'] = mmcv_runner_mod
mmcv_runner_base_module_mod = types.ModuleType('mmcv.runner.base_module')
sys.modules['mmcv.runner.base_module'] = mmcv_runner_base_module_mod
import mmcv.utils as mmcv_utils_mod
sys.modules['mmcv.utils'] = mmcv_utils_mod
mmcv_utils_registry_mod = types.ModuleType('mmcv.utils.registry')
sys.modules['mmcv.utils.registry'] = mmcv_utils_registry_mod

# legacy collate function ----------------------------------------------------
def _legacy_collate(batch, samples_per_gpu=None):  # type: ignore
    """Delegate to the original MMCV1 `collate` implementation.

    Args:
        batch (Sequence): A list of data samples.
        samples_per_gpu (int | None): Per-GPU batch size for padding logic.
    Returns:
        Any: Batched data following MMCV1 semantics.
    """
    from legacy.collate import collate as _legacy_mmcv_collate
    from mmengine.dist import get_dist_info
    
    if samples_per_gpu is None:
        samples_per_gpu = 1
    
    # For single GPU training, we need to handle the data differently
    rank, world_size = get_dist_info()
    if world_size == 1:  # Single GPU case
        # Use samples_per_gpu = len(batch) to avoid grouping
        result = _legacy_mmcv_collate(batch, samples_per_gpu=len(batch))
        
        # Unwrap DataContainer for stack=True items in single GPU case
        if hasattr(result, '__class__') and result.__class__.__name__ == 'DataContainer':
            if result.stack and isinstance(result.data, list) and len(result.data) == 1:
                # Return the single tensor directly instead of a list
                result._data = result.data[0]
        elif isinstance(result, dict):
            # Handle dict of DataContainers
            for key, value in result.items():
                if hasattr(value, '__class__') and value.__class__.__name__ == 'DataContainer':
                    if value.stack and isinstance(value.data, list) and len(value.data) == 1:
                        value._data = value.data[0]
        
        return result
    else:
        # Multi-GPU case, use original logic
        return _legacy_mmcv_collate(batch, samples_per_gpu=samples_per_gpu)

mmcv_parallel_mod.collate = _legacy_collate  # type: ignore

from mmdet.registry import DATASETS as _MMDET_DATASETS
def _legacy_build_dataset(cfg, default_args=None):
    return _MMDET_DATASETS.build(cfg, default_args=default_args)

import mmdet.datasets as mmdet_datasets_mod
sys.modules['mmdet.datasets'] = mmdet_datasets_mod
mmdet_datasets_mod.build_dataset = _legacy_build_dataset
mmdet_datasets_mod.DATASETS = _MMDET_DATASETS  # type: ignore

import mmdet3d.datasets as mmdet3d_datasets_mod
sys.modules['mmdet3d.datasets'] = mmdet3d_datasets_mod
mmdet3d_datasets_mod.build_dataset = _legacy_build_dataset
mmdet3d_datasets_mod.DATASETS = _MMDET_DATASETS  # type: ignore

from plugin.datasets.builder import build_dataloader as _plugin_build_dataloader
mmdet_datasets_mod.build_dataloader = _plugin_build_dataloader  # type: ignore
mmdet3d_datasets_mod.build_dataloader = _plugin_build_dataloader  # type: ignore

from mmengine.registry import TRANSFORMS as _PIPELINES
from mmdet.registry import DATASETS as _DATASETS
mmdet_datasets_builder_mod = types.ModuleType('mmdet.datasets.builder')
mmdet_datasets_builder_mod.PIPELINES = _PIPELINES
mmdet_datasets_builder_mod.DATASETS = _DATASETS
sys.modules['mmdet.datasets.builder'] = mmdet_datasets_builder_mod

import torch
from mmengine.model.wrappers.distributed import MMDistributedDataParallel as _MMDDP

from legacy.scatter_gather import scatter as _legacy_scatter, scatter_kwargs as _legacy_scatter_kwargs
mmcv_parallel_mod.scatter = _legacy_scatter  # type: ignore
mmcv_parallel_mod.scatter_kwargs = _legacy_scatter_kwargs  # type: ignore

from mmengine.registry import HOOKS as _MME_HOOKS, RUNNERS as _MME_RUNNERS
mmcv_runner_mod.HOOKS = _MME_HOOKS  # type: ignore
mmcv_runner_mod.RUNNERS = _MME_RUNNERS  # type: ignore

# Checkpoint I/O ------------------------------------------------------------
from mmengine.runner.checkpoint import load_checkpoint as _load_ckpt  # type: ignore
mmcv_runner_mod.load_checkpoint = _load_ckpt  # type: ignore

# FP16 wrappers -------------------------------------------------------------
def _wrap_fp16_model(model):  # type: ignore
    """A very light replacement of mmcv.runner.wrap_fp16_model.

    It simply converts the model to half precision if possible. While more
    sophisticated logic exists in the original implementation (e.g. handling
    BN layers), this minimal variant is usually good enough for inference.
    """
    try:
        import torch
        model.half()
    except Exception:
        pass
    return model

mmcv_runner_mod.wrap_fp16_model = _wrap_fp16_model  # type: ignore

def auto_fp16(apply_to=None):  # type: ignore
    """Legacy decorator stub that leaves the function unchanged."""
    def _decorator(func):
        return func
    return _decorator

mmcv_runner_mod.auto_fp16 = auto_fp16  # type: ignore

# Hook related aliases -------------------------------------------------------
from mmengine.hooks import Hook as _BaseHook
from mmengine.hooks import DistSamplerSeedHook as _DSSH  # type: ignore
mmcv_runner_mod.DistSamplerSeedHook = _DSSH  # type: ignore

# Provide aliases so that type checks like `isinstance(runner, EpochBasedRunner)` work.
from mmengine.runner import Runner as _MMEngineRunner  # noqa: F401
class _EpochBasedRunner(_MMEngineRunner):  # type: ignore
    pass
class _IterBasedRunner(_MMEngineRunner):  # type: ignore
    pass
mmcv_runner_mod.EpochBasedRunner = _EpochBasedRunner  # type: ignore
mmcv_runner_mod.IterBasedRunner = _IterBasedRunner  # type: ignore

from mmengine.registry import build_from_cfg as _build_from_cfg_alias
mmcv_utils_mod.build_from_cfg = _build_from_cfg_alias  # type: ignore

# Provide Registry alias and mmcv.utils.registry submodule -------------------
from mmengine.registry import Registry as _RegistryAlias, build_from_cfg as _build_from_cfg_alias  # type: ignore

# Create mmcv.utils.registry submodule
mmcv_utils_registry_mod.Registry = _RegistryAlias  # type: ignore
mmcv_utils_registry_mod.build_from_cfg = _build_from_cfg_alias  # type: ignore

# Expose registry submodule in mmcv.utils
mmcv_utils_mod.registry = mmcv_utils_registry_mod  # type: ignore
# Also expose Registry directly in mmcv.utils for backward compatibility
mmcv_utils_mod.Registry = _RegistryAlias  # type: ignore

import mmdet.datasets.samplers as _mmdet_ds_samplers_mod

def _build_dataset(cfg, default_args=None):
    return _DATASETS.build(cfg, default_args=default_args)
mmdet_datasets_builder_mod.build_dataset = _build_dataset

# ------------- proactively import custom pipeline modules ------------------

def _import_submodules(package_name: str):
    """Recursively import all submodules under a package."""
    import importlib, pkgutil, sys
    try:
        pkg = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return
    if not hasattr(pkg, '__path__'):
        return
    for _loader, _name, _is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        if _name not in sys.modules:
            importlib.import_module(_name)

# ---------- legacy shim for mmdet3d.apis.single_gpu_test --------------------
try:
    _m3d_apis_mod = importlib.import_module('mmdet3d.apis')
except ModuleNotFoundError:
    _m3d_apis_mod = types.ModuleType('mmdet3d.apis')

sys.modules['mmdet3d.apis'] = _m3d_apis_mod

if not hasattr(_m3d_apis_mod, 'single_gpu_test'):
    def _single_gpu_test(model, data_loader, show=False, out_dir=None):  # type: ignore
        """A minimal re-implementation of the deprecated
        ``mmdet3d.apis.single_gpu_test`` API.

        Args:
            model (nn.Module): The model to be tested (already on correct GPU).
            data_loader (DataLoader): The dataloader providing input batches.
            show (bool): Unused here, kept for compatibility.
            out_dir (str | None): Optional directory to save visualization.

        Returns:
            list | dict: Raw predictions accumulated over the dataset.
        """
        import torch, tqdm, os
        model.eval()
        results = []
        prog_bar = None
        try:
            from mmcv import ProgressBar  # type: ignore
            prog_bar = ProgressBar(len(data_loader.dataset))
        except Exception:
            prog_bar = None
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    # legacy datasets sometimes return (inputs, data_samples)
                    kwargs = data[1] if isinstance(data[1], dict) else {}
                    inputs = data[0]
                    batch_result = model(inputs, **kwargs)
                else:
                    batch_result = model(return_loss=False, rescale=True, **data)
                # Assume batch size = 1 for 3D tasks; if not, extend accordingly.
                if isinstance(batch_result, dict) and 'bbox_results' in batch_result:
                    results.extend(batch_result['bbox_results'])
                else:
                    # generic append
                    if isinstance(batch_result, (list, tuple)):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                if prog_bar is not None:
                    if isinstance(batch_result, (list, tuple)):
                        prog_bar.update(len(batch_result))
                    else:
                        prog_bar.update()
                # optional visualization (best-effort)
                if show or out_dir is not None:
                    try:
                        from mmdet.visualization import Det3DLocalVisualizer  # type: ignore
                        visualizer = Det3DLocalVisualizer(out_dir=out_dir)  # type: ignore
                        visualizer.add_datasample('result', data, batch_result)
                    except Exception:
                        pass  # silently ignore visualisation if deps missing
        return results

    _m3d_apis_mod.single_gpu_test = _single_gpu_test  # type: ignore

# ---------- legacy build_model shim for mmdet3d.models & mmdet.models --------
from mmengine.registry import MODELS as _MME_MODELS


def _legacy_build_model(cfg, train_cfg=None, test_cfg=None):  # type: ignore
    """Wrapper to replicate the removed `build_model` helper.

    Args:
        cfg (dict): Model config.
        train_cfg (dict | None): Training config to inject.
        test_cfg (dict | None): Test config to inject.

    Returns:
        nn.Module: The built model instance.
    """
    if train_cfg is not None and 'train_cfg' not in cfg:
        cfg['train_cfg'] = train_cfg
    if test_cfg is not None and 'test_cfg' not in cfg:
        cfg['test_cfg'] = test_cfg
    from mmdet.registry import MODELS as _MMDET_MODELS
    return _MMDET_MODELS.build(cfg)
    # return _MME_MODELS.build(cfg)

for _models_module_name in ('mmdet3d.models', 'mmdet.models'):
    try:
        _models_mod = importlib.import_module(_models_module_name)
    except ModuleNotFoundError:
        _models_mod = types.ModuleType(_models_module_name)
        sys.modules[_models_module_name] = _models_mod
    if not hasattr(_models_mod, 'build_model'):
        _models_mod.build_model = _legacy_build_model  # type: ignore

# ---------------------------------------------------------------------------

# ---------- legacy helper replace_ImageToTensor -----------------------------
from copy import deepcopy as _deepcopy

def _replace_ImageToTensor(pipeline):  # type: ignore
    """Recursively replace the legacy 'ImageToTensor' transform with
    'DefaultFormatBundle' in a data pipeline config.

    This mirrors the helper provided in MMDet v2.x.
    """
    if pipeline is None:
        return None

    def _process(seq):
        processed = []
        for trans in seq:
            t = _deepcopy(trans)
            if isinstance(t, dict):
                if t.get('type') == 'ImageToTensor':
                    t = dict(type='DefaultFormatBundle')
                else:
                    # handle nested pipelines (e.g., MultiScaleFlipAug)
                    for k in ('transforms', 'pipeline', 'ops'):
                        if k in t and isinstance(t[k], list):
                            t[k] = _process(t[k])
            processed.append(t)
        return processed

    return _process(pipeline)

for _ds_mod_name in ('mmdet.datasets', 'mmdet3d.datasets'):
    try:
        _ds_mod = importlib.import_module(_ds_mod_name)
    except ModuleNotFoundError:
        _ds_mod = types.ModuleType(_ds_mod_name)
        sys.modules[_ds_mod_name] = _ds_mod
    if not hasattr(_ds_mod, 'replace_ImageToTensor'):
        _ds_mod.replace_ImageToTensor = _replace_ImageToTensor  # type: ignore

# ---------- legacy get_root_logger in mmdet.utils ---------------------------
try:
    _mmdet_utils_mod = importlib.import_module('mmdet.utils')
except ModuleNotFoundError:
    _mmdet_utils_mod = types.ModuleType('mmdet.utils')
    sys.modules['mmdet.utils'] = _mmdet_utils_mod

if not hasattr(_mmdet_utils_mod, 'get_root_logger'):
    from mmengine.logging import MMLogger as _MME_Logger  # type: ignore

    def _get_root_logger(log_file=None, log_level='INFO'):
        """Return a root MMLogger instance (compat)."""
        # Try to fetch current; if none, create new one.
        logger = _MME_Logger.get_current_instance()
        if logger is None:
            logger = _MME_Logger.get_instance(name='root', log_file=log_file, log_level=log_level)
        return logger

    _mmdet_utils_mod.get_root_logger = _get_root_logger  # type: ignore

# Ensure legacy Collect3D and Transformer are registered (requires xavier_init first)
import legacy.formatting  # noqa: F401  # side-effect registration
import legacy.transformer  # noqa: F401  # ensure legacy transformer classes registered

warnings.filterwarnings("ignore")