import os


class Config:
    def __init__(self):
        self.orig_input_folder = r"/media/npu-x/Seagate Expansion Drive/acrobat_validation_pyramid_1_of_1"
        self.target_intput_folder = r'./input/acrobat_validation_pyramid_1_of_1_sorted'


if __name__ == "__main__":
    config = Config()
    image_names = [_ for _ in os.listdir(config.orig_input_folder) if
                   os.path.isfile(os.path.join(config.orig_input_folder, _))]

    image_ids = set()
    for image_name in image_names:
        image_id = image_name.split("_")[0]
        image_ids.add(image_id)

    count = 0
    for image_id in image_ids:
        new_path = os.path.join(config.target_intput_folder, image_id)
        os.makedirs(new_path, exist_ok=True)
        image_names_by_id = [image_name for image_name in image_names if image_name.split("_")[0] == image_id]
        count += len(image_names_by_id)
        for image_name in image_names_by_id:
            src_path = os.path.join(config.orig_input_folder, image_name)
            dst_path = os.path.join(config.target_intput_folder, image_id, image_name)
            os.symlink(src_path, dst_path)
