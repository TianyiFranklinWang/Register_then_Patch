import contextlib
import datetime
import gc
import os
import sys
import time
from itertools import product

import cv2
import numpy as np
import skimage
import tifffile as tiff
import torch

from DeeperHistReg.create_acrobat_submission import register_ones


class Config:
    def __init__(self):
        self.register = True
        self.registration_input_folder = "./input/acrobat_train_pyramid_sorted"
        self.registration_output_folder = "/media/npu-x/DataOne2T/acrobat_train_pyramid_processed/registered"
        self.registration_down_sample_rate = 4
        self.save_registration = True

        self.patch = True
        self.patch_input_folder = self.registration_output_folder
        self.patch_output_folder = "/media/npu-x/DataOne2T/acrobat_train_pyramid_processed/patched"
        self.patch_down_sample_rate = None
        self.patch_size = 256
        self.pad_value_wsi = 255
        self.thresh_method = 'otsu'  # or 'adaptive'
        self.area_threshold = 16384
        self.min_size = 16384
        self.connectivity = 8
        self.save_format = 'png'
        self.save_patch = True

        self.target_image_modality = "HE"

        self.debug = False


@contextlib.contextmanager
def mute_stdout(debug=False):
    if not debug:
        class DummyFile:
            def write(self, x): pass

        old_stdout = sys.stdout
        sys.stdout = DummyFile()
        try:
            yield
        finally:
            sys.stdout = old_stdout
    else:
        yield


class MultiModalDataset:
    def __init__(self, root):
        self.root = root
        self.image_ids = [_ for _ in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, _))]
        self.image_dict, self.modalities = self.gather_images()

    def gather_images(self):
        image_dict = dict()
        modalities = set()
        for image_id in self.image_ids:
            current_folder = os.path.join(self.root, image_id)
            image_dict[image_id] = [_ for _ in os.listdir(current_folder) if
                                    os.path.isfile(os.path.join(current_folder, _))]
            for image_name in image_dict[image_id]:
                modalities.add(self.get_modality(image_name))

        return image_dict, modalities

    @staticmethod
    def get_modality(image_name):
        return image_name.split(".")[0].split('_')[1]

    def __len__(self):
        count = 0
        for image_names in self.image_dict.values():
            count += len(image_names)
        return count


def resize(image, scale_factor):
    if isinstance(image, np.ndarray):
        original_size = image.shape
        target_size = (original_size[1] // scale_factor, original_size[0] // scale_factor)
        return cv2.resize(image, dsize=target_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    elif isinstance(image, torch.Tensor):
        original_size = image.shape
        target_size = (original_size[2] // scale_factor, original_size[3] // scale_factor)
        return torch.nn.functional.interpolate(image, size=target_size, mode='area')


def pad_wsi(wsi: np.ndarray, pad_size: int, pad_value: int) -> np.ndarray:
    """Pad the wsi in order to be dividable.
    Args:
        wsi (np.ndarray): WSI to be pad.
        pad_size (int): WSI will be padded to the integer multiples of pad_size.
        pad_value (int): Padding value.
    Returns:
        np.ndarray: Padded wsi.
    """
    scaled_shape = wsi.shape
    pad0, pad1 = (int(pad_size - (scaled_shape[0] % pad_size)),
                  int(pad_size - (scaled_shape[1] % pad_size)))
    if len(scaled_shape) == 3:
        wsi = np.pad(wsi, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                     constant_values=pad_value)
    elif len(scaled_shape) == 2:
        wsi = np.pad(wsi, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]],
                     constant_values=pad_value)
    return wsi


def thresh_wsi(config: Config, wsi: np.ndarray) -> np.ndarray:
    """Apply thresholding to the wsi.
    Args:
        config (Config): Configurations.
        wsi (np.ndarray): Wsi to be threshed.
    Returns:
        np.ndarray: Threshed wsi.
    """
    gray_scaled_wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2GRAY)
    blured_scaled_wsi = cv2.medianBlur(gray_scaled_wsi, 3)
    if config.thresh_method == "adaptive":
        threshed_wsi = cv2.adaptiveThreshold(blured_scaled_wsi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 21, 8)
    elif config.thresh_method == "otsu":
        _, threshed_wsi = cv2.threshold(blured_scaled_wsi, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    else:
        raise AttributeError(f"No thresh method named {config.thresh_method}")
    threshed_wsi = skimage.morphology.remove_small_holes(threshed_wsi > 0, area_threshold=config.area_threshold,
                                                         connectivity=config.connectivity)
    threshed_wsi = skimage.morphology.remove_small_objects(threshed_wsi, min_size=config.min_size,
                                                           connectivity=config.connectivity)
    return threshed_wsi.astype(np.uint8) * 255


def gen_patch(wsi: np.ndarray, patch_size: int) -> np.ndarray:
    """Generate Patches from wsi of given size.
    Args:
        wsi (np.ndarray): Wsi to be processed.
        patch_size (int): Size of a patch.
    Returns:
        np.ndarray: Patches of given size.
    """
    shape = wsi.shape
    if len(shape) == 2:
        patches = wsi.reshape(shape[0] // patch_size, patch_size,
                              shape[1] // patch_size, patch_size)
        patches = patches.transpose(0, 2, 1, 3)
        patches = patches.reshape(-1, patch_size, patch_size)
    elif len(shape) == 3:
        patches = wsi.reshape(shape[0] // patch_size, patch_size,
                              shape[1] // patch_size, patch_size, 3)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, patch_size, patch_size, 3)
    return patches


def select_patch(thresh_patches: np.ndarray) -> list[int]:
    """Select out patches that contain information.
    Args:
        thresh_patches (np.ndarray): Patches from threshed wsi.
    Returns:
        list[int]: List of selected indices.
    """
    selected_idx = list()
    for idx, thresh_patch in enumerate(thresh_patches):
        if thresh_patch.sum() > 0:
            selected_idx.append(idx)
    return selected_idx


def fetch_coordinates(selected_idx, wsi_shape, patch_size):
    x_patch_num = int(wsi_shape[1] / patch_size)
    y_patch_num = int(wsi_shape[0] / patch_size)
    coordinates_dict = dict()
    for x_idx, y_idx in product(range(x_patch_num), range(y_patch_num)):
        idx = x_idx + y_idx * x_patch_num
        if idx in selected_idx:
            coordinates_dict[idx] = (x_idx * patch_size, y_idx * patch_size)
    return coordinates_dict


def pick_source_and_target(image_names, target_image_modality):
    target_image_name = None
    source_image_names = list()
    for image_name in image_names:
        if target_image_modality == MultiModalDataset.get_modality(image_name):
            if target_image_name is None:
                target_image_name = image_name
            else:
                raise AttributeError(
                    f"Conflict in target images: '{target_image_name}' '{image_name}'")
        else:
            source_image_names.append(image_name)
    return target_image_name, source_image_names


def register_protocol(config):
    print("\n--------- Initializing Registration Protocol ---------")
    print("    -> Creating MultiModalDataset")
    registration_dataset = MultiModalDataset(root=config.registration_input_folder)
    print(f"        - Total images: {len(registration_dataset)}")

    print("\n--------- Executing Registration Protocol ---------")
    os.makedirs(config.registration_output_folder, exist_ok=True)

    print(f"    -> Processing on {len(registration_dataset.image_ids)} batches of images")
    for image_id in registration_dataset.image_ids:
        print(f"\n        - Processing on Image ID: {image_id}")

        image_names = registration_dataset.image_dict[image_id]
        target_image_name, source_image_names = pick_source_and_target(image_names, config.target_image_modality)
        print(f"            - Target Image: {target_image_name}")
        print(f"            - Source Images: {source_image_names}")

        os.makedirs(os.path.join(config.registration_output_folder, image_id), exist_ok=True)

        print(f"                - Processing on {target_image_name}")
        target_path = os.path.join(registration_dataset.root, image_id, target_image_name)
        target_image = tiff.imread(target_path)
        target_image = resize(target_image, scale_factor=config.registration_down_sample_rate)
        if config.save_registration:
            tiff.imwrite(os.path.join(config.registration_output_folder, image_id, target_image_name), target_image)
        del target_image
        gc.collect()

        for source_image_name in source_image_names:
            print(f"                - Processing on {source_image_name}")
            source_path = os.path.join(registration_dataset.root, image_id, source_image_name)
            with mute_stdout(debug=config.debug):
                warped_source = register_ones(target_path=target_path, source_path=source_path,
                                              down_sample_rate=config.registration_down_sample_rate, device='cuda:0')
            if config.save_registration:
                tiff.imwrite(os.path.join(config.registration_output_folder, image_id, source_image_name),
                             warped_source)
            del warped_source
            gc.collect()


def patch_protocol(config):
    print("\n--------- Initializing Patch Protocol ---------")
    print("    -> Creating MultiModalDataset")
    patch_dataset = MultiModalDataset(root=config.patch_input_folder)
    print(f"        - Total images: {len(patch_dataset)}")

    print("\n--------- Executing Patch Protocol ---------")
    os.makedirs(config.patch_output_folder, exist_ok=True)
    for modality in patch_dataset.modalities:
        modality_output_folder = os.path.join(config.patch_output_folder, modality)
        os.makedirs(modality_output_folder, exist_ok=True)

    print(f"    -> Processing on {len(patch_dataset.image_ids)} batches of images")
    for image_id in patch_dataset.image_ids:
        print(f"\n        - Processing on Image ID: {image_id}")

        image_names = patch_dataset.image_dict[image_id]
        target_image_name, source_image_names = pick_source_and_target(image_names, config.target_image_modality)
        print(f"            - Target Image: {target_image_name}")
        print(f"            - Source Images: {source_image_names}")

        print(f"                - Processing on {target_image_name}")
        target_path = os.path.join(patch_dataset.root, image_id, target_image_name)
        target_image = tiff.imread(target_path)
        if config.patch_down_sample_rate is not None:
            target_image = resize(target_image, scale_factor=config.patch_down_sample_rate)
        target_image = pad_wsi(target_image, config.patch_size, config.pad_value_wsi)

        threshed_wsi = thresh_wsi(config, target_image)
        threshed_patches = gen_patch(threshed_wsi, config.patch_size)
        selected_idx = select_patch(threshed_patches)
        coordinates_dict = fetch_coordinates(selected_idx, threshed_wsi.shape, config.patch_size)
        del threshed_wsi
        del threshed_patches
        gc.collect()

        target_image_patches = gen_patch(target_image, config.patch_size)
        if config.save_patch:
            for idx in selected_idx:
                coordinate = coordinates_dict[idx]
                save_patch_name = f"{image_id}_{coordinate[0]}_{coordinate[1]}.{config.save_format}"
                save_path = os.path.join(config.patch_output_folder, config.target_image_modality, save_patch_name)
                patch = target_image_patches[idx]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, patch)
        del target_image
        del target_image_patches
        gc.collect()

        for source_image_name in source_image_names:
            print(f"                - Processing on {source_image_name}")
            source_path = os.path.join(patch_dataset.root, image_id, source_image_name)
            source_image = tiff.imread(source_path)
            source_image = pad_wsi(source_image, config.patch_size, config.pad_value_wsi)
            source_image_patches = gen_patch(source_image, config.patch_size)

            if config.save_patch:
                for idx in selected_idx:
                    coordinate = coordinates_dict[idx]
                    save_patch_name = f"{image_id}_{coordinate[0]}_{coordinate[1]}.{config.save_format}"
                    save_path = os.path.join(config.patch_output_folder, patch_dataset.get_modality(source_image_name),
                                             save_patch_name)
                    patch = source_image_patches[idx]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, patch)
            del source_image
            del source_image_patches
            gc.collect()


def main(config):
    print("\n--------- Executing Multi-modality Preprocess Protocol ---------")
    start_time = time.time()
    if config.register:
        register_protocol(config)
    if config.patch:
        patch_protocol(config)
    print(f"--------- Total time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))} ---------")


if __name__ == "__main__":
    config = Config()
    main(config)
