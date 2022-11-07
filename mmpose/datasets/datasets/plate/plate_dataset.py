# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class PlateDataset(BaseCocoStyleDataset):
    """Plate dataset"""

    METAINFO: dict = dict(from_file='configs/_base_/datasets/plate.py')
