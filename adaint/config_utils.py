"""Configuration helpers shared by the MMEditing train and test entrypoints."""

import os.path as osp

import mmcv


def _normalize_pipeline_scales(value):
    """Old MMEditing Resize requires tuples, whereas YAML has only lists."""
    if isinstance(value, dict):
        if value.get('type') == 'Resize' and isinstance(value.get('scale'), list):
            value['scale'] = tuple(value['scale'])
        for child in value.values():
            _normalize_pipeline_scales(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _normalize_pipeline_scales(child)


def load_config(filename):
    """Load the legacy Python config or a YAML config into ``mmcv.Config``.

    MMEditing's original ``Config.fromfile`` only handles Python configs in
    the version bundled with this repository.  YAML is deliberately converted
    to the same Config object, so the runner, datasets, and registries keep
    their existing behaviour.
    """
    if not isinstance(filename, str):
        return filename
    if osp.splitext(filename)[1].lower() not in {'.yaml', '.yml'}:
        cfg = mmcv.Config.fromfile(filename)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                'YAML configs require PyYAML; install it in the training environment.') from exc
        with open(filename, 'r') as handle:
            cfg = mmcv.Config(yaml.safe_load(handle) or {}, filename=filename)

        # YAML has no native tuple syntax when parsed with safe_load.  MMCV's
        # IterBasedRunner requires every workflow item to be a tuple, matching
        # the legacy Python configs: workflow = [('train', 1)].
        if cfg.get('workflow', None) is not None:
            cfg.workflow = [
                tuple(item) if isinstance(item, list) else item
                for item in cfg.workflow
            ]

    _normalize_pipeline_scales(cfg._cfg_dict)
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)
    from .context_transform import configure_context_pipeline
    configure_context_pipeline(cfg)
    return cfg
