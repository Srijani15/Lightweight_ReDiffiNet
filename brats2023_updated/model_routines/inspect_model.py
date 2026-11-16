"""Print the configurable baseline U-Net architecture and parameter count.

Run from the directory above the package, for example:
``python -m brats2023_updated.model_routines.inspect_model --base-channels 32``.
"""

from __future__ import annotations

import argparse

from ..models.unet3d import U_Net3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-channels', type=int, default=U_Net3d.ORIGINAL_BASE_CHANNELS)
    args = parser.parse_args()
    model = U_Net3d(base_channels=args.base_channels)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    fp32_memory_mib = total_parameters * 4 / 1024 ** 2
    fp16_memory_mib = total_parameters * 2 / 1024 ** 2

    print('Architecture: brats2023_updated.models.unet3d.U_Net3d')
    print(f'First encoder channel width: {model.base_channels}')
    print(f'Encoder channel widths: {model.channel_widths}')
    print(f'Output channels (WT/TC/ET): {model.output_ch}')
    print(f'Total parameters: {total_parameters:,}')
    print(f'Trainable parameters: {trainable_parameters:,}')
    print(f'Parameter memory only: {fp32_memory_mib:.2f} MiB FP32; {fp16_memory_mib:.2f} MiB FP16')


if __name__ == '__main__':
    main()
