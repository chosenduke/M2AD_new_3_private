import numpy as np


def m2ad_fuse(obj_names, cls_names,
             illumination_list, view_list,
             imgs_masks, image_anomalys, object_anomalys,
             anomaly_scores, anomaly_maps,
             fuse='add', **kwargs):
    """
    Fuse instances with the same object_name, cls_name, and view under different illuminations to a single instance.

    Parameters:
    - obj_names: List of object names corresponding to each instance.
    - cls_names: List of class names corresponding to each instance.
    - illumination_list: List of illumination conditions for each instance.
    - view_list: List of view views corresponding to each instance.
    - imgs_masks: List of image masks (H x W) for each instance.
    - image_anomalys: List of image anomaly values (scalar for each instance).
    - object_anomalys: List of object anomaly values (scalar for each instance).
    - anomaly_scores: List of anomaly scores (scalar for each instance).
    - anomaly_maps: List of anomaly maps (H x W for each instance).
    - fuse: Fusion method ('add' or 'mul').

    Returns:
    - fused results
    """

    # Initialize variables to store the fused results
    view_instance_masks = []
    view_instance_anomalys = []

    view_instance_anomaly_maps = []
    view_instance_anomaly_scores = []

    view_instance_names = [] # object name + view
    view_instance_cls_names = []
    
    object_instance_names = []
    object_instance_cls_names = []
    object_instance_anomalys = []
    object_instance_anomaly_scores = []

    # Loop over the unique combinations of (obj_name, cls_name, view)
    unique_combinations = set(zip(obj_names, cls_names, view_list, object_anomalys))

    for obj_name, cls_name, view, object_anomaly in unique_combinations:
        # Filter indices where the obj_name, cls_name, and view match
        indices = [i for i, (o, c, v, a) in enumerate(zip(obj_names, cls_names, view_list, object_anomalys)) if
                   o == obj_name and c == cls_name and v == view and a == object_anomaly]

        view_instance_names.append(np.array([f'{obj_name}_{view}']))
        view_instance_cls_names.append(np.array([cls_name]))

        # Fuse image masks (H x W): use the maximum value at each spatial location
        masks = [imgs_masks[i] for i in indices]
        fused_masks = np.maximum.reduce(masks)
        view_instance_masks.append(fused_masks)

        # Fuse image anomaly values: use the maximum anomaly value across instances
        image_anomalys_values = [image_anomalys[i] for i in indices]
        if len(set(image_anomalys_values)) > 1:
            print(f"Error: Inconsistent object anomaly values for {obj_name}, {cls_name}, {view}")
        fused_image_anomalys = np.maximum.reduce(image_anomalys_values)
        view_instance_anomalys.append(np.array(fused_image_anomalys)[np.newaxis])

        # Fuse anomaly scores: either add or multiply
        scores = [anomaly_scores[i] for i in indices]
        if fuse == 'add':
            fused_anomaly_scores = np.mean(scores)
        elif fuse == 'mul':
            product = np.prod(scores)
            fused_anomaly_scores = product ** (1 / len(scores))
        else:
            raise NotImplementedError

        view_instance_anomaly_scores.append(np.array(fused_anomaly_scores)[np.newaxis])

        # Fuse anomaly maps (H x W): use the maximum value at each spatial location
        maps = [anomaly_maps[i] for i in indices]
        if fuse == 'add':
            fused_anomaly_maps = np.mean(maps, axis=0)
        elif fuse == 'mul':
            product = np.prod(maps, axis=0)
            fused_anomaly_maps = product ** (1 / len(maps))
        else:
            raise NotImplementedError

        view_instance_anomaly_maps.append([fused_anomaly_maps])

    # Loop over the unique combinations of (obj_name, cls_name, view)
    unique_combinations = set(zip(obj_names, cls_names, object_anomalys))

    for obj_name, cls_name, object_anomaly in unique_combinations:
        # Filter indices where the obj_name, cls_name, and view match
        indices = [i for i, (o, c, a) in enumerate(zip(obj_names, cls_names, object_anomalys)) if
                   o == obj_name and c == cls_name and a == object_anomaly]

        object_instance_names.append(np.array([obj_name]))
        object_instance_cls_names.append(np.array([cls_name]))

        # Fuse anomaly scores: either add or multiply
        scores = [anomaly_scores[i] for i in indices]
        if fuse == 'add':
            fused_anomaly_scores = np.mean(scores)
        elif fuse == 'mul':
            product = np.prod(scores)
            fused_anomaly_scores = product ** (1 / len(scores))
        else:
            raise NotImplementedError

        object_instance_anomaly_scores.append(np.array(fused_anomaly_scores)[np.newaxis])

        # Fuse image anomaly values: use the maximum anomaly value across instances
        object_anomalys_values = [object_anomalys[i] for i in indices]
        if len(set(object_anomalys_values)) > 1:
            print(f"Error: Inconsistent object anomaly values for {obj_name}, {cls_name}")
        fused_image_anomalys = object_anomalys_values[0]
        object_instance_anomalys.append(np.array(fused_image_anomalys)[np.newaxis])

    results = dict(
        view_instance_masks=view_instance_masks,
        view_instance_anomalys=view_instance_anomalys,
        view_instance_anomaly_maps=view_instance_anomaly_maps,
        view_instance_anomaly_scores=view_instance_anomaly_scores,
        view_instance_names=view_instance_names,
        view_instance_cls_names=view_instance_cls_names,
        object_instance_names=object_instance_names,
        object_instance_anomalys=object_instance_anomalys,
        object_instance_anomaly_scores=object_instance_anomaly_scores,
        object_instance_cls_names=object_instance_cls_names,
    )

    results = {k: np.concatenate(v, axis=0) for k, v in results.items()}

    return results
