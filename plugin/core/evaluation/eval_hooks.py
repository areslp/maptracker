# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
# 从 legacy 目录导入正确的基类实现
from legacy.evaluation import DistEvalHook as BaseDistEvalHook, EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

# Register hook into MMEngine registry so it can be referenced by
# config files using ``type='CustomDistEvalHook'``.
from mmdet.registry import HOOKS


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    """Utility to compute dynamic evaluation intervals.

    Args:
        start_interval (int): Initial evaluation interval.
        dynamic_interval_list (list[tuple]): Each tuple is (milestone, interval).

    Returns:
        tuple: milestones list, intervals list corresponding to milestones.
    """
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals

@HOOKS.register_module()
class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        # Build the dataloader in case a config dict / ConfigDict is passed in.
        # New versions of MMEngine pass ``dataloader`` as a config dictionary
        # while the legacy ``BaseDistEvalHook`` expects a real
        # ``torch.utils.data.DataLoader`` instance.
        dataloader_cfg = kwargs.get('dataloader', None)
        if dataloader_cfg is not None:
            from torch.utils.data import DataLoader  # local import to avoid circular deps
            # If a DataLoader instance is not provided, build it using MMEngine
            if not isinstance(dataloader_cfg, DataLoader):
                try:
                    # ``Runner.build_dataloader`` can convert a dataloader cfg
                    # dict/ConfigDict into a pytorch ``DataLoader``.
                    from mmengine.runner import Runner
                    kwargs['dataloader'] = Runner.build_dataloader(dataloader_cfg)
                except Exception as e:
                    raise TypeError(
                        'Failed to build DataLoader from cfg for CustomDistEvalHook. '
                        'Original error: {}'.format(e)
                    )

        # Call the parent constructor with the processed kwargs
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner, **kwargs):
        """Evaluate the model only at the start of training by epoch.

        Accept **kwargs so that it is compatible with runner.call_hook that may
        forward additional information such as ``batch_idx`` or
        ``data_batch`` which we do not use here.
        """
        self._decide_interval(runner)
        super().before_train_epoch(runner, **kwargs)

    def before_train_iter(self, runner, **kwargs):
        """Evaluate periodically during iteration-based training.

        Extra keyword arguments are ignored to maintain compatibility with
        other hooks that expect them.
        """
        self._decide_interval(runner)
        super().before_train_iter(runner, **kwargs)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from ..apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
  
