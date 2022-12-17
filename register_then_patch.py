import contextlib
import os
import sys

import cv2
import numpy as np
import tifffile as tiff
import torch

from DeeperHistReg.create_acrobat_submission import register_ones


class Config:
    def __init__(self):
        self.input_folder = "./input"

        self.registration_output_folder = "./output/registered"
        self.target_image_modality = "HE"
        self.down_sample_rate = 4

        self.debug = False
        self.save = False


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
        return image_name.split('_')[1]

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


def main(config):
    print("\n--------- Executing Multi-modality Preprocess Protocol ---------")

    print("\n--------- Initializing Registration Protocol ---------")
    print("    -> Creating MultiModalDataset")
    dataset = MultiModalDataset(root=config.input_folder)
    print(f"        - Total images: {len(dataset)}")

    print("\n--------- Executing Registration Protocol ---------")
    os.makedirs(config.registration_output_folder, exist_ok=True)
    print(f"    -> Processing on {len(dataset.image_ids)} batches of images")
    for image_id in dataset.image_ids:
        print(f"\n        - Processing on Image ID: {image_id}")

        image_names = dataset.image_dict[image_id]
        target_image_name = None
        source_image_names = list()
        for image_name in image_names:
            if config.target_image_modality == dataset.get_modality(image_name):
                if target_image_name is None:
                    target_image_name = image_name
                else:
                    raise AttributeError(
                        f"Conflict in target images: {image_id}    '{target_image_name}' '{image_name}'")
            else:
                source_image_names.append(image_name)
        print(f"            - Target Image: {target_image_name}")
        print(f"            - Source Images: {source_image_names}")

        os.makedirs(os.path.join(config.registration_output_folder, image_id), exist_ok=True)

        print(f"                - Processing on {target_image_name}")
        target_path = os.path.join(dataset.root, image_id, target_image_name)
        target_image = tiff.imread(target_path)
        target_image = resize(target_image, scale_factor=config.down_sample_rate)
        if config.save:
            tiff.imwrite(os.path.join(config.registration_output_folder, image_id, target_image_name), target_image)

        for source_image_name in source_image_names:
            print(f"                - Processing on {source_image_name}")
            source_path = os.path.join(dataset.root, image_id, source_image_name)
            with mute_stdout(debug=config.debug):
                warped_source = register_ones(target_path=target_path, source_path=source_path,
                                              down_sample_rate=config.down_sample_rate, device='cuda:0')
            if config.save:
                tiff.imwrite(os.path.join(config.registration_output_folder, image_id, source_image_name),
                             warped_source)


if __name__ == "__main__":
    config = Config()
    main(config)
