# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
#
# Pipeline code derived from HuggingFace Diffusers (Apache 2.0).
# Original Qwen-Image pipeline: Copyright 2025 Qwen Team.

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class QwenEditPipelineOutput(BaseOutput):
    """
    Output class for Qwen-Edit pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape 
            `(batch_size, height, width, num_channels)`.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]
