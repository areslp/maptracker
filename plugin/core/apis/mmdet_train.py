# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import random
import warnings
import pprint


import numpy as np
import torch
import torch.distributed as dist
from mmengine.model.wrappers import MMDistributedDataParallel
from torch.nn import DataParallel as MMDataParallel
from legacy.scatter_gather import scatter as _legacy_scatter
from mmcv.runner import (HOOKS, IterBasedRunner)
from mmengine.dist import get_dist_info
from mmengine.registry import RUNNERS
from mmengine.registry import build_from_cfg

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from ...datasets.builder import build_dataloader
from ..evaluation.eval_hooks import CustomDistEvalHook


@RUNNERS.register_module()
class MyRunnerWrapper(IterBasedRunner):

    def __init__(self, *args, **kwargs):
        """Initialize MyRunnerWrapper with necessary attributes for hooks."""
        super().__init__(*args, **kwargs)
        
        # Ensure work_dir is properly set for CheckpointHook
        if not hasattr(self, 'work_dir') or self.work_dir is None:
            self.work_dir = kwargs.get('work_dir', './work_dir')
        
        # Ensure other attributes that CheckpointHook might need
        if not hasattr(self, 'logger'):
            self.logger = kwargs.get('logger', None)
            
        # Initialize log_buffer if it doesn't exist (needed for evaluation hooks)
        if not hasattr(self, 'log_buffer'):
            # Create a simple log buffer with an output dict (needed by eval hooks)
            # 使用 legacy 目录中的完整 LogBuffer 实现
            from legacy.log_buffer import LogBuffer
            self.log_buffer = LogBuffer()
            
        # Initialize rank attribute for distributed training
        if not hasattr(self, 'rank'):
            try:
                import torch.distributed as dist
                self.rank = dist.get_rank() if dist.is_initialized() else 0
            except:
                self.rank = 0

    @property
    def iter(self):
        """Expose current iteration count for hooks."""
        return getattr(self, '_iter', 0)
    
    @property
    def epoch(self):
        """Expose current epoch count for hooks."""
        return getattr(self, '_epoch', 0)
    
    def train(self, data_loader, **kwargs):
        """One training iteration.

        Handles DataLoader iteration safely, keeps track of epoch/iter counters
        and calls the necessary hooks expected by MMCV/MMEngine. Validation is
        triggered by EvalHook that listens to ``after_train_iter``.
        """

        # Lazily create an iterator over the dataloader and reuse it until
        # exhausted.
        if not hasattr(self, '_data_iter'):
            self._data_iter = iter(data_loader)

        try:
            data_batch = next(self._data_iter)
        except StopIteration:
            # ------------------ End of epoch ------------------
            # Call hooks signalling the end of current epoch
            self.call_hook('after_train_epoch')

            # Update epoch counter
            if not hasattr(self, '_epoch'):
                self._epoch = 0  # type: ignore
            self._epoch += 1  # type: ignore

            # Inform sampler (for distributed shuffling)
            if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
                data_loader.sampler.set_epoch(self._epoch)

            # Reset iterator & inner-epoch counter
            self._inner_iter = 0  # reset inner-epoch iteration counter
            self._data_iter = iter(data_loader)

            # Call hooks signalling the beginning of new epoch
            self.call_hook('before_train_epoch')

            data_batch = next(self._data_iter)

        # Ensure counters exist
        if not hasattr(self, '_iter'):
            self._iter = 0  # type: ignore
        if not hasattr(self, '_inner_iter'):
            self._inner_iter = 0  # type: ignore

        # --------------------------------------------------------------
        # Unwrap Distributed/DataParallel so that both single-GPU (raw
        # model) and multi-GPU (model.module) cases are supported.
        # --------------------------------------------------------------
        inner_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Expose counters for the model (custom logic)
        inner_model.num_iter = self._iter
        inner_model.num_epoch = getattr(self, '_epoch', 0)

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader

        # Unwrap DataContainer & move tensors to current GPU
        _cur_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        data_batch = _legacy_scatter(data_batch, [_cur_dev])[0]
        print(f"data_batch type: {type(data_batch)}")

        if self._iter == 0 and self.rank == 0:  # 只在第一迭代、rank0 打印
            for k, v in data_batch.items():
                if isinstance(v, (list, tuple)):
                    print(f"{k}: {type(v)} len={len(v)} -> inner type {type(v[0])}")
                else:
                    print(f"{k}: {type(v)}")

        # ------------------------------------------------------------------
        # MMEngine DistributedModelWrapper expects ``model.module`` to have a
        # ``data_preprocessor`` attr.  Legacy MMDet v2 models do not define
        # this.  Provide a no-op implementation on-the-fly for compatibility.
        # ------------------------------------------------------------------
        if not hasattr(inner_model, 'data_preprocessor'):
            class _NoopDataPreprocessor(torch.nn.Module):
                def forward(self, data, training=False):  # type: ignore
                    return data

            inner_model.data_preprocessor = _NoopDataPreprocessor().to(_cur_dev)  # type: ignore

        # ------------------------------------------------------------------
        # Legacy MMDet v2 models output a dict of losses but do not implement
        # ``parse_losses`` expected by MMEngine wrappers.  Provide a fallback
        # implementation that mimics the old behaviour.
        # ------------------------------------------------------------------
        if not hasattr(inner_model, 'parse_losses'):
            import types

            def _default_parse_losses(self, losses):  # type: ignore
                """Parse the raw outputs (loss dict) and compute total loss & log_vars."""
                # Case 1: Model already returns (loss, log_vars, *rest)
                if isinstance(losses, tuple):
                    assert len(losses) >= 2, 'Expect (loss, log_vars, ...)'
                    total_loss, log_vars = losses[0], losses[1]
                    if isinstance(total_loss, torch.Tensor):
                        total_loss = total_loss.mean()
                    elif isinstance(total_loss, (int, float)):
                        total_loss = torch.tensor(total_loss, device=next(self.parameters()).device)  # type: ignore

                    # Ensure log_vars are python scalars for logging
                    log_vars = {k: (v.mean().item() if isinstance(v, torch.Tensor) else float(v)) for k, v in log_vars.items()}
                    return total_loss, log_vars

                # Case 2: losses is a dict as in MMDet v3 style
                if isinstance(losses, dict):
                    log_vars = {}
                    total_loss = 0.0
                    for name, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            loss_value = value.mean()
                        elif isinstance(value, list):
                            loss_value = sum(v.mean() for v in value)
                        else:
                            raise TypeError(f'Unsupported loss value type {type(value)} for {name}')

                        log_vars[name] = loss_value
                        if 'loss' in name:
                            total_loss += loss_value

                    log_vars['loss'] = total_loss

                    # Convert tensors to python scalars for logging
                    log_vars = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in log_vars.items()}

                    return total_loss, log_vars

                raise TypeError(f'Unsupported losses type {type(losses)}')

            # Bind as method to the model instance
            inner_model.parse_losses = types.MethodType(_default_parse_losses, inner_model)  # type: ignore

        # ----- Hooks & train step -----
        self.call_hook('before_train_iter', batch_idx=self._inner_iter, data_batch=data_batch)
        outputs = self.model.train_step(data_batch, self.optim_wrapper, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs
        self.call_hook('after_train_iter', batch_idx=self._inner_iter, data_batch=data_batch, outputs=outputs)

        # Update counters
        self._inner_iter += 1  # type: ignore
        self._iter += 1  # type: ignore

    # ------------------------------------------------------------------
    # Legacy MMCV compat: provide a ``run`` method that executes training
    # according to ``workflow``.  Most configs in this repo use
    # ``workflow = [('train', 1)]`` which means running training for the
    # entire iteration budget.  We therefore implement a minimal version
    # that handles this common case; other phases (val/test) are ignored.
    # ------------------------------------------------------------------
    def run(self, data_loaders, workflow, **kwargs):  # type: ignore
        """Execute the workflow.

        Args:
            data_loaders (list): list of dataloaders, the first element is
                used for training.
            workflow (list[tuple]): e.g. [('train', 1)]. Only 'train' phase
                is respected in this lightweight wrapper.
        """
        assert isinstance(data_loaders, (list, tuple)) and len(data_loaders) > 0, \
            'data_loaders must be non-empty list/tuple'
        assert isinstance(workflow, list) and len(workflow) > 0, \
            'workflow must be non-empty list'

        # Currently only support patterns like [('train', N)] where N>0
        phase, _ = workflow[0]
        if phase != 'train':
            raise NotImplementedError('MyRunnerWrapper only supports train workflow')

        train_loader = data_loaders[0]

        # Initialise counters at the beginning of training
        self._iter = 0  # type: ignore
        self._epoch = 0  # type: ignore

        # Call run-start & epoch-start hooks
        self.call_hook('before_run')
        self.call_hook('before_train')
        self.call_hook('before_train_epoch')

        # Only support pure train workflow; validation occurs via EvalHook.
        try:
            while self._iter < self.max_iters:  # type: ignore
                self.train(train_loader)
        finally:
            # Ensure run-end hooks are executed even if an error occurs
            self.call_hook('after_run')


def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # ---- Build Runner via MMEngine Registry ----
    runner = RUNNERS.build(cfg)
    print(f"runner type: {type(runner)}")

    # Preserve eval_model reference if provided (avoid messing with Runner.__init__ signature)
    if eval_model is not None:
        runner.eval_model = eval_model

    # ---------------- Debug optimizer / optim_wrapper -----------------
    def wrapper_info(w):
        """Return a dict of all attributes except the heavy `optimizer` one."""
        return {k: v for k, v in w.__dict__.items() if k != 'optimizer'}

    # ------------------------------------------------------------------
    # Ensure `optim_wrapper` and `param_schedulers` are fully built **before**
    # the training loop starts.  In the vanilla MMEngine workflow this is
    # done inside `Runner.train()`, but our custom training pipeline skips
    # that call, so we replicate the essential steps here.
    # ------------------------------------------------------------------
    from mmengine.config import ConfigDict

    if isinstance(runner.optim_wrapper, (dict, ConfigDict)):
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)  # type: ignore

        # Auto-scale learning rate if the config requests it.
        if hasattr(runner, 'auto_scale_lr'):
            runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)  # type: ignore

    if runner.param_schedulers is not None and isinstance(
            runner.param_schedulers, (dict, list, ConfigDict)):
        runner.param_schedulers = runner.build_param_scheduler(runner.param_schedulers)  # type: ignore

    # Initialise the internal counters of OptimWrapper (needed for grad-accum etc.)
    if hasattr(runner.optim_wrapper, 'initialize_count_status'):
        runner.optim_wrapper.initialize_count_status(runner.model, 0, runner.max_iters)  # type: ignore

    print('\n=== DEBUG: Runner Optimizer Info ===')
    # 1) runner.optim_wrapper：MMEngine 推荐的封装
    print('runner.optim_wrapper ->', type(runner.optim_wrapper))
    pprint.pprint(wrapper_info(runner.optim_wrapper))


    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        raise ValueError("fp16_cfg is not supported in this codebase")

    # 打印所有已注册的hooks信息（放在最后，确保包含所有hooks）
    if getattr(runner, 'rank', 0) == 0:  # 多卡时只让 rank0 打印
        print('\n===== DEBUG: ALL REGISTERED HOOKS =====')
        eval_hook_found = False
        for idx, hook in enumerate(runner.hooks):
            hook_type = hook.__class__.__name__
            priority = hook.priority
            interval = getattr(hook, 'interval', 'N/A')
            by_epoch = getattr(hook, 'by_epoch', 'N/A')
            print(f'[{idx:02d}] {hook_type:25s}  interval={str(interval):<6}  by_epoch={str(by_epoch):<5}  priority={priority}')
            if 'Eval' in hook_type:
                eval_hook_found = True
        if eval_hook_found:
            print("✓ Eval hook found in registered hooks!")
        else:
            print("✗ No eval hook found in registered hooks!")
        print('========================================')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    data_loaders = [runner.train_dataloader]
    runner.run(data_loaders, cfg.workflow)

