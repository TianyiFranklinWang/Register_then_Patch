### Python Imports ###
import copy
import gc
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import time

### External Imports ###
import numpy as np
import torch as tc
import cv2

### Internal Imports ###
import DeeperHistReg.preprocessing as pre
import DeeperHistReg.postprocessing as pst
import DeeperHistReg.utils as u
import DeeperHistReg.utils_tc as utc

import DeeperHistReg.initial_registration as ir
import DeeperHistReg.nonrigid_registration as nr
import DeeperHistReg.io_affine as ioa

import DeeperHistReg.acrobat_submission_configs as configs


def gather_cases(input_path):
    cases = os.listdir(input_path)
    cases_dict = dict()
    for case in cases:
        cases_dict[case] = os.listdir(os.path.join(input_path, case))
    return cases_dict


def split_images(image_names, target_image_type='HE'):
    target_image_name = None
    source_image_names = list()
    for image_name in image_names:
        stain_type = image_name.split('_')[1]
        if stain_type == target_image_type:
            target_image_name = image_name
        else:
            source_image_names.append(image_name)
    if target_image_name is None:
        raise RuntimeError(f"Must have at least one target image of type {target_image_type}, found none")
    return target_image_name, source_image_names


def resize(image, scale_factor):
    if isinstance(image, np.ndarray):
        original_size = image.shape
        target_size = (original_size[1] // scale_factor, original_size[0] // scale_factor)
        return cv2.resize(image, dsize=target_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    elif isinstance(image, torch.Tensor):
        original_size = image.shape
        target_size = (original_size[2] // scale_factor, original_size[3] // scale_factor)
        return torch.nn.functional.interpolate(image, size=target_size, mode='area')


def create_acrobat_submission(**config):
    input_datapath = config['input_datapath']
    output_path = config['output_path']
    level = config['level']
    registration_method = getattr(sys.modules[__name__], config['registration_method'])
    registration_params = config['registration_params']
    preprocessing_params = config['preprocessing_params']

    #############################

    cases_dict = gather_cases(input_datapath)
    target_paths = list()
    source_paths = list()
    for case, image_names in cases_dict.items():
        target_image_name, source_image_names = split_images(image_names, target_image_type='HE')
        for source_image_name in source_image_names:
            target_paths.append(os.path.join(case, target_image_name))
            source_paths.append(os.path.join(case, source_image_name))

    # source_paths = natsort.natsorted(
    #     [item.replace('.ndpi', '.tiff') for item in pd.unique(input_dataframe['anon_filename_he'])])
    # target_paths = natsort.natsorted(
    #     [item.replace('.ndpi', '.tiff') for item in pd.unique(input_dataframe['anon_filename_ihc'])])

    # Registration Params
    device = "cuda:0"

    # Actual Registration
    for i in range(len(source_paths)):
        tc.cuda.empty_cache()
        ### Load Images
        source_path = source_paths[i]
        target_path = target_paths[i]
        print(f"Source path: {source_path}")
        print(f"Target path: {target_path}")
        source, source_slide = u.load_slide(os.path.join(input_datapath, source_path), level, load_slide=True)
        target, target_slide = u.load_slide(os.path.join(input_datapath, target_path), level, load_slide=True)
        print(f"Source dimensions: {source_slide.level_dimensions}")
        print(f"Target dimensions: {target_slide.level_dimensions}")

        source = resize(source, scale_factor=4)
        target = resize(target, scale_factor=4)

        source = u.image_to_tensor(source, device)
        target = u.image_to_tensor(target, device)

        ### Preprocessing
        print(f"Original source shape: {source.shape}")
        print(f"Original target shape: {target.shape}")
        preprocessing_function = pre.get_function(preprocessing_params['preprocessing_function'])
        pre_source, pre_target, _, _, postprocessing_params = preprocessing_function(source, target, None, None,
                                                                                     preprocessing_params)
        print(f"Preprocessed source shape: {pre_source.shape}")
        print(f"Preprocessed target shape: {pre_target.shape}")

        ### Perform Registration
        displacement_field = registration_method(pre_source, pre_target, **registration_params)

        ### Save Visual Results
        warped_source = u.warp_image(pre_source, displacement_field)
        case_id = source_path.split("_")[1]
        output_case_path = os.path.join(output_path, str(case_id))
        if not os.path.isdir(output_case_path):
            os.makedirs(output_case_path)
        u.save_image(pre_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(),
                     os.path.join(output_case_path, 'source.jpg'),
                     renormalize=True)
        u.save_image(pre_target[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(),
                     os.path.join(output_case_path, 'target.jpg'),
                     renormalize=True)
        u.save_image(warped_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy(),
                     os.path.join(output_case_path, 'warped_source.jpg'), renormalize=True)

        ### Update CSV File
        # displacement_field_np = u.tc_df_to_np_df(displacement_field)
        # update_landmarks(pre_source, pre_target, source, target, input_dataframe, output_dataframe, source_path,
        #                  displacement_field_np, level, postprocessing_params)
        tc.cuda.empty_cache()

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # output_dataframe.to_csv(output_csv_path)
    return


def update_landmarks(pre_source, pre_target, source, target, input_dataframe, output_dataframe, image_path,
                     displacement_field, level, preprocessing_params):
    case_id = image_path.split("_")[0]
    landmarks_x = input_dataframe[input_dataframe.anon_id == int(case_id)]['ihc_x']
    landmarks_y = input_dataframe[input_dataframe.anon_id == int(case_id)]['ihc_y']
    lx = np.array(landmarks_x, dtype=np.float32)
    ly = np.array(landmarks_y, dtype=np.float32)
    landmarks = np.stack((lx, ly), axis=-1)
    scaler = input_dataframe[input_dataframe.anon_id == int(case_id)]['mpp_ihc_10X'] * 2 ** level
    landmarks[:, 0] /= scaler
    landmarks[:, 1] /= scaler
    landmarks = pre.target_landmarks_preprocessing(landmarks, preprocessing_params)
    warped_landmarks = u.warp_landmarks(landmarks, displacement_field)
    warped_landmarks = pst.target_landmarks_postprocessing(warped_landmarks, preprocessing_params)
    warped_landmarks[:, 0] *= scaler
    warped_landmarks[:, 1] *= scaler
    output_dataframe.loc[output_dataframe.anon_id == int(case_id), 'he_x'] = warped_landmarks[:, 0]
    output_dataframe.loc[output_dataframe.anon_id == int(case_id), 'he_y'] = warped_landmarks[:, 1]


### Methods ###

def affine_iterative(source, target, **config):
    tc.cuda.empty_cache()
    ### Initial Alignment ###
    affine_params = config['affine_params']
    b_t = time.time()
    # BEWARE - REVERSED
    transform = ir.rotated_landmark_based_combination(target, source, affine_params)
    e_t = time.time()
    print(f"Elapsed time: {e_t - b_t}")
    final_transform = tc.eye(3)
    final_transform[0:2, 0:3] = transform
    final_transform = tc.linalg.inv(final_transform)
    transform = final_transform[0:2, 0:3].unsqueeze(0).to("cuda:0")

    iterative_affine_params = config['iterative_affine_params']
    iterative_transform = ioa.instance_optimization_affine_registration(source, target, transform,
                                                                        iterative_affine_params)
    transforms = [transform, iterative_transform]

    best_cost = np.inf
    best_transform = transform
    resampled_source, resampled_target = u.initial_resampling(source, target, 256)
    # sift = cv2.xfeatures2d.SIFT_create(256)  # 256
    sift = cv2.SIFT_create(256)
    for tr in transforms:
        displacement_field = utc.tc_transform_to_tc_df(tr, resampled_source.size())
        warped_source = u.warp_image(resampled_source, displacement_field)
        keypoints, source_descriptors = sift.detectAndCompute(
            (warped_source[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), None)
        _, target_descriptors = sift.compute(
            (resampled_target[0, 0, :, :].detach().cpu().numpy() * 255).astype(np.uint8), keypoints)
        try:
            costs = np.mean((source_descriptors - target_descriptors) ** 2, axis=1)
            lowest_costs = np.sort(costs)[0:8]
            current_cost = np.mean(lowest_costs)
        except:
            current_cost = np.inf
        if current_cost < best_cost:
            best_cost = current_cost
            best_transform = tr
    displacement_field_ini = utc.tc_transform_to_tc_df(best_transform, source.size())
    return displacement_field_ini


def affine_iterative_nonrigid(source, target, **config):
    tc.cuda.empty_cache()
    ### Initial Alignment ###
    displacement_field_ini = affine_iterative(source, target, **config)
    tc.cuda.empty_cache()

    ### Nonrigid Registration ###
    nonrigid_params = config['nonrigid_params']
    displacement_field_nr = nr.instance_optimization_nonrigid_registration(source, target, displacement_field_ini,
                                                                           nonrigid_params)
    tc.cuda.empty_cache()
    return displacement_field_nr


def run():
    config = configs.affine_nonrigid_config()
    create_acrobat_submission(**config)
    pass


def register_ones(target_path, source_path, down_sample_rate=4, device='cuda:0'):
    config = configs.affine_nonrigid_config()

    level = config['level']
    registration_method = getattr(sys.modules[__name__], config['registration_method'])
    registration_params = config['registration_params']
    preprocessing_params = config['preprocessing_params']

    #############################

    # Actual Registration

    tc.cuda.empty_cache()
    ### Load Images
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")
    source, source_slide = u.load_slide(source_path, level, load_slide=True)
    target, target_slide = u.load_slide(target_path, level, load_slide=True)
    print(f"Source dimensions: {source_slide.level_dimensions}")
    print(f"Target dimensions: {target_slide.level_dimensions}")

    source = resize(source, scale_factor=down_sample_rate)
    target = resize(target, scale_factor=down_sample_rate)
    source = cv2.resize(source, (target.shape[1], target.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    source = u.image_to_tensor(source, device)
    target = u.image_to_tensor(target, device)
    warped_source = copy.deepcopy(source)

    ### Preprocessing
    print(f"Original source shape: {source.shape}")
    print(f"Original target shape: {target.shape}")
    preprocessing_function = pre.get_function(preprocessing_params['preprocessing_function'])
    pre_source, pre_target, _, _, postprocessing_params = preprocessing_function(source, target, None, None,
                                                                                 preprocessing_params)
    print(f"Preprocessed source shape: {pre_source.shape}")
    print(f"Preprocessed target shape: {pre_target.shape}")

    ### Perform Registration
    displacement_field = registration_method(pre_source, pre_target, **registration_params)

    ### Save Visual Results
    # warped_source = u.warp_image(pre_source, displacement_field)
    warped_source = u.warp_image(warped_source, displacement_field)

    warped_source = warped_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
    if warped_source.shape[2] == 3:
        warped_source = (warped_source * 255).astype(np.uint8)
    elif warped_source.shape[2] == 1:
        warped_source = (warped_source[:, :, 0] * 255)
        warped_source = warped_source.astype(np.uint8)
    else:
        raise ValueError("Unsupported image format.")

    tc.cuda.empty_cache()

    return warped_source


def register_ones_no_downsample(target_path, source_path, down_sample_rate=4, device='cuda:0'):
    config = configs.affine_nonrigid_config()

    level = config['level']
    registration_method = getattr(sys.modules[__name__], config['registration_method'])
    registration_params = config['registration_params']
    preprocessing_params = config['preprocessing_params']

    #############################

    # Actual Registration

    tc.cuda.empty_cache()
    ### Load Images
    print(f"Source path: {source_path}")
    print(f"Target path: {target_path}")
    source, source_slide = u.load_slide(source_path, level, load_slide=True)
    target, target_slide = u.load_slide(target_path, level, load_slide=True)
    print(f"Source dimensions: {source_slide.level_dimensions}")
    print(f"Target dimensions: {target_slide.level_dimensions}")

    orig_target_shape = target.shape[:2]

    source = resize(source, scale_factor=down_sample_rate)
    target = resize(target, scale_factor=down_sample_rate)
    source = cv2.resize(source, (target.shape[1], target.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    source = u.image_to_tensor(source, device)
    target = u.image_to_tensor(target, device)

    ### Preprocessing
    print(f"Original source shape: {source.shape}")
    print(f"Original target shape: {target.shape}")
    preprocessing_function = pre.get_function(preprocessing_params['preprocessing_function'])
    pre_source, pre_target, _, _, postprocessing_params = preprocessing_function(source, target, None, None,
                                                                                 preprocessing_params)
    print(f"Preprocessed source shape: {pre_source.shape}")
    print(f"Preprocessed target shape: {pre_target.shape}")

    ### Perform Registration
    displacement_field = registration_method(pre_source, pre_target, **registration_params)

    del source, target, pre_source, pre_target
    gc.collect()
    source, source_slide = u.load_slide(source_path, level, load_slide=True)
    source = cv2.resize(source, (orig_target_shape[1], orig_target_shape[0]), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    source = u.image_to_tensor(source, device='cpu')

    displacement_field = displacement_field.detach().cpu().numpy().squeeze()
    displacement_field_0 = cv2.resize(displacement_field[:, :, 0], (orig_target_shape[1], orig_target_shape[0]), fx=0,
                                      fy=0, interpolation=cv2.INTER_CUBIC)
    displacement_field_1 = cv2.resize(displacement_field[:, :, 1], (orig_target_shape[1], orig_target_shape[0]), fx=0,
                                      fy=0, interpolation=cv2.INTER_CUBIC)
    displacement_field_0 = np.expand_dims(displacement_field_0, axis=2)
    displacement_field_1 = np.expand_dims(displacement_field_1, axis=2)
    displacement_field = np.concatenate([displacement_field_0, displacement_field_1], axis=2)
    displacement_field = np.expand_dims(displacement_field, axis=0)
    displacement_field = torch.from_numpy(displacement_field)

    ### Save Visual Results
    # warped_source = u.warp_image(pre_source, displacement_field)
    warped_source = u.warp_image(source, displacement_field)

    warped_source = warped_source[0, :, :, :].detach().cpu().permute(1, 2, 0).numpy()
    if warped_source.shape[2] == 3:
        warped_source = (warped_source * 255).astype(np.uint8)
    elif warped_source.shape[2] == 1:
        warped_source = (warped_source[:, :, 0] * 255)
        warped_source = warped_source.astype(np.uint8)
    else:
        raise ValueError("Unsupported image format.")

    tc.cuda.empty_cache()

    return warped_source


if __name__ == "__main__":
    run()
