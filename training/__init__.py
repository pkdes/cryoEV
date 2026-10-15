"""Training package.

The U-Net re-exports below are LAZY. Importing them eagerly pulled in
segmentation_models_pytorch at `import training`, which meant the YOLO
inference path could not run without the entire legacy U-Net stack
installed. Same public names, resolved on first attribute access.
"""

_UNET_EXPORTS = ('create_model', 'train_model', 'CombinedLoss', 'get_loss_function')

__all__ = list(_UNET_EXPORTS)


def __getattr__(name):
    if name in _UNET_EXPORTS:
        from . import train_unet
        return getattr(train_unet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
