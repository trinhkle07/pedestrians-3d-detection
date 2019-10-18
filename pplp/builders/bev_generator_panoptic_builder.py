from pplp.core.bev_generators import bev_panoptic_slices


def build(bev_maps_type_config, panoptic_utils):

    bev_maps_type = bev_maps_type_config.WhichOneof('bev_maps_type')

    if bev_maps_type == 'slices':
        return bev_panoptic_slices.BevSlices(
            bev_maps_type_config.slices, panoptic_utils)

    raise ValueError('Invalid bev_maps_type', bev_maps_type)
