from copy import deepcopy

from google.protobuf import text_format

import pplp
from pplp.datasets.panoptic.panoptic_dataset import PanopticDataset
from pplp.protos import panoptic_dataset_pb2
from pplp.protos.panoptic_dataset_pb2 import PanopticDatasetConfig


class DatasetBuilder(object):
    """
    Static class to return preconfigured dataset objects
    """

    PANOPTIC_UNITTEST = PanopticDatasetConfig(
        name="unittest-panoptic",
        dataset_dir=pplp.root_dir() + "/tests/datasets/Panoptic/object",
        data_split="train",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )

    PANOPTIC_TRAIN = PanopticDatasetConfig(
        name="panoptic",
        data_split="train",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1]
    )

    PANOPTIC_VAL = PanopticDatasetConfig(
        name="panoptic",
        data_split="val",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )

    PANOPTIC_TEST = PanopticDatasetConfig(
        name="panoptic",
        data_split="test",
        data_split_dir="testing",
        has_labels=False,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )

    PANOPTIC_TRAINVAL = PanopticDatasetConfig(
        name="panoptic",
        data_split="trainval",
        data_split_dir="training",
        has_labels=True,
        cluster_split="trainval",
        classes=["Pedestrian"],
        num_clusters=[1],
    )

    PANOPTIC_TRAIN_MINI = PanopticDatasetConfig(
        name="panoptic",
        data_split="train_mini",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )
    PANOPTIC_VAL_MINI = PanopticDatasetConfig(
        name="panoptic",
        data_split="val_mini",
        data_split_dir="training",
        has_labels=True,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )
    PANOPTIC_TEST_MINI = PanopticDatasetConfig(
        name="panoptic",
        data_split="test_mini",
        data_split_dir="testing",
        has_labels=False,
        cluster_split="train",
        classes=["Pedestrian"],
        num_clusters=[1],
    )

    CONFIG_DEFAULTS_PROTO = \
        """
        bev_source: 'lidar'

        panoptic_utils_config {
            area_extents: [-4, 4, -5, 3, 0, 7]
            voxel_size: 0.01
            anchor_strides: [0.5, 0.5]

            bev_generator {
                slices {
                    height_lo: -0.2
                    height_hi: 2.3
                    num_slices: 1
                }
            }

            mini_batch_config {
                density_threshold: 1

                rpn_config {
                    iou_2d_thresholds {
                        neg_iou_lo: 0.0
                        neg_iou_hi: 0.3
                        pos_iou_lo: 0.5
                        pos_iou_hi: 1.0
                    }
                    # iou_3d_thresholds {
                    #     neg_iou_lo: 0.0
                    #     neg_iou_hi: 0.005
                    #     pos_iou_lo: 0.1
                    #     pos_iou_hi: 1.0
                    # }

                    mini_batch_size: 512
                }

                avod_config {
                    iou_2d_thresholds {
                        neg_iou_lo: 0.0
                        neg_iou_hi: 0.55
                        pos_iou_lo: 0.65
                        pos_iou_hi: 1.0
                    }

                    mini_batch_size: 1024
                }
            }
        }
        """

    @staticmethod
    def load_dataset_from_config(dataset_config_path):

        dataset_config = panoptic_dataset_pb2.PanopticDatasetConfig()
        with open(dataset_config_path, 'r') as f:
            text_format.Merge(f.read(), dataset_config)

        return DatasetBuilder.build_panoptic_dataset(dataset_config,
                                                  use_defaults=False)

    @staticmethod
    def copy_config(cfg):
        return deepcopy(cfg)

    @staticmethod
    def merge_defaults(cfg):
        cfg_copy = DatasetBuilder.copy_config(cfg)
        text_format.Merge(DatasetBuilder.CONFIG_DEFAULTS_PROTO, cfg_copy)
        return cfg_copy

    @staticmethod
    def build_panoptic_dataset(base_cfg,
                            use_defaults=True,
                            new_cfg=None) -> PanopticDataset:
        """Builds a PanopticDataset object using the provided configurations

        Args:
            base_cfg: a base dataset configuration
            use_defaults: whether to use the default config values
            new_cfg: (optional) a custom dataset configuration, no default
                values will be used, all config values must be provided

        Returns:
            PanopticDataset object
        """
        cfg_copy = DatasetBuilder.copy_config(base_cfg)

        if use_defaults:
            # Use default values
            text_format.Merge(DatasetBuilder.CONFIG_DEFAULTS_PROTO, cfg_copy)

        if new_cfg:
            # Use new config values if provided
            cfg_copy.MergeFrom(new_cfg)

        return PanopticDataset(cfg_copy)


def main():
    DatasetBuilder.build_panoptic_dataset(DatasetBuilder.PANOPTIC_TRAIN_MINI)


if __name__ == '__main__':
    main()
