import bisect
import math
import os
import random
import open3d

from PIL import Image
import cv2
import numpy as np
import oxford.options
import torch

from models.multimodal_classifier import MMClassifer, MMClassiferCoarse

from data.kitti_helper import camera_matrix_scaling, camera_matrix_cropping, FarthestSampler


def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance


def get_camera_timestamp(pc_timestamp_idx,
                         traversal_pc_num,
                         pc_timestamps_list,
                         pc_poses_np,
                         camera_timestamps_list,
                         camera_poses_np):
    translation_max = test_translation_max = 10.0
    # pc is built every opt.pc_build_interval (2m),
    # so search for the previous/nex pc_timestamp that > max_translation
    index_interval = math.ceil(translation_max / 2)

    previous_pc_t_idx = max(0, pc_timestamp_idx - index_interval)
    previous_pc_t = pc_timestamps_list[previous_pc_t_idx]
    next_pc_t_idx = min(traversal_pc_num - 1, pc_timestamp_idx + index_interval)
    next_pc_t = pc_timestamps_list[next_pc_t_idx]

    previous_cam_t_idx = bisect.bisect_left(camera_timestamps_list, previous_pc_t)
    next_cam_t_idx = bisect.bisect_left(camera_timestamps_list, next_pc_t)

    P_o_pc = pc_poses_np[pc_timestamp_idx]
    while True:
        cam_t_idx = random.randint(previous_cam_t_idx, next_cam_t_idx)
        P_o_cam = camera_poses_np[cam_t_idx]
        P_cam_pc = np.dot(np.linalg.inv(P_o_cam), P_o_pc)
        t_norm = np.linalg.norm(P_cam_pc[0:3, 3])

        if t_norm < translation_max:
            break

    return cam_t_idx, P_cam_pc


def downsample_np(pc_np, intensity_np, k):
    if pc_np.shape[1] >= k:
        choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
    else:
        fix_idx = np.asarray(range(pc_np.shape[1]))
        while pc_np.shape[1] + fix_idx.shape[0] < k:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
        random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
    pc_np = pc_np[:, choice_idx]
    intensity_np = intensity_np[:, choice_idx]

    return pc_np, intensity_np


if __name__ == "__main__":
    opt = oxford.options.Options()
    farthest_sampler = FarthestSampler(dim=3)
    K = np.array([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)

    device = torch.device("cpu")
    # data_process =

    print("=> using '{}' for computation.".format(device))
    root_path = '/data/oxford'
    # -------------------- create model --------------------
    print("=> creating model and optimizer... ")
    if opt.is_fine_resolution:
        model = MMClassifer(opt, writer=None)
    else:
        model = MMClassiferCoarse(opt, writer=None)

    model_path = os.path.join(root_path, 'checkpoints/best.pth')
    print(model_path)
    model.load_model(model_path)
    model.detector.eval()

    # -------------------- load data --------------------
    traversal = "2015-11-10-10-32-52"
    dataset = []
    pc_timestamps_list_dict = {}
    pc_poses_np_dict = {}
    camera_timestamps_list_dict = {}
    camera_poses_np_dict = {}

    pc_timestamps_np = np.load(os.path.join(root_path, traversal, 'pc_timestamps.npy'))
    print("pc_timestamps")
    print(pc_timestamps_np.shape)
    print(pc_timestamps_np[:10])
    pc_timestamps_list_dict[traversal] = pc_timestamps_np.tolist()
    pc_poses_np = np.load(os.path.join(root_path, traversal, 'pc_poses.npy')).astype(np.float32)
    print("pc_poses")
    print(pc_poses_np[:10])
    # convert it to camera coordinate
    P_convert = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    P_convert_inv = np.linalg.inv(P_convert)
    for b in range(pc_poses_np.shape[0]):
        pc_poses_np[b] = np.dot(P_convert, np.dot(pc_poses_np[b], P_convert_inv))

    pc_poses_np_dict[traversal] = pc_poses_np

    img_timestamps_np = np.load(os.path.join(root_path, traversal, 'camera_timestamps.npy'))
    print("img_timestamps")
    print(img_timestamps_np.shape)
    print(img_timestamps_np[:10])
    camera_timestamps_list_dict[traversal] = img_timestamps_np.tolist()
    img_poses_np = np.load(os.path.join(root_path, traversal, 'camera_poses.npy')).astype(np.float32)
    print("img_poses")
    print(img_poses_np[:10])
    # convert it to camera coordinate
    for b in range(img_poses_np.shape[0]):
        img_poses_np[b] = np.dot(P_convert, np.dot(img_poses_np[b], P_convert_inv))

    camera_poses_np_dict[traversal] = img_poses_np

    for i in range(pc_timestamps_np.shape[0]):
        pc_timestamp = pc_timestamps_np[i]
        dataset.append((traversal, pc_timestamp, i, pc_timestamps_np.shape[0]))

    # -------------------- prepare data --------------------

    index = 0
    traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num = dataset[index]
    pc_timestamps_list = pc_timestamps_list_dict[traversal]
    pc_poses_np = pc_poses_np_dict[traversal]
    camera_timestamps_list = camera_timestamps_list_dict[traversal]
    camera_poses_np = camera_poses_np_dict[traversal]

    camera_timestamp_idx, P_cam_pc = get_camera_timestamp(pc_timestamp_idx,
                                                          traversal_pc_num,
                                                          pc_timestamps_list,
                                                          pc_poses_np,
                                                          camera_timestamps_list,
                                                          camera_poses_np)

    camera_folder = os.path.join(root_path, traversal, 'stereo', 'centre')
    print(f"camera folder = {camera_folder}")
    camera_timestamp = camera_timestamps_list[camera_timestamp_idx]
    print(f"camera image = {os.path.join(camera_folder, '%d.jpg' % camera_timestamp)}")
    img = np.array(Image.open(os.path.join(camera_folder, "%d.jpg" % camera_timestamp)))
    print(img.shape)
    # ------------- load image, original size is 960x1280, bottom rows are car itself -------------
    tmp_img_H = img.shape[0]
    img = img[0:(tmp_img_H - opt.crop_original_bottom_rows), :, :]
    # scale
    img = cv2.resize(img,
                     (int(round(img.shape[1] * opt.img_scale)), int(round((img.shape[0] * opt.img_scale)))),
                     interpolation=cv2.INTER_LINEAR)
    K = camera_matrix_scaling(K, opt.img_scale)

    # random crop into input size
    img_crop_dx = int((img.shape[1] - opt.img_W) / 2)
    img_crop_dy = int((img.shape[0] - opt.img_H) / 2)
    # crop image
    img = img[img_crop_dy:img_crop_dy + opt.img_H,
          img_crop_dx:img_crop_dx + opt.img_W, :]
    K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

    # ------------- load point cloud ----------------
    if opt.is_remove_ground:
        lidar_name = 'lms_front_foreground'
    else:
        lidar_name = 'lms_front'
    pc_path = os.path.join(root_path, traversal, lidar_name, '%d.npy' % pc_timestamp)
    npy_data = np.load(pc_path).astype(np.float32)
    # shuffle the point cloud data, this is necessary!
    npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
    pc_np = npy_data[0:3, :]  # 3xN
    intensity_np = npy_data[3:4, :]  # 1xN

    # limit max_z, the pc is in CAMERA coordinate
    pc_np_x_square = np.square(pc_np[0, :])
    pc_np_z_square = np.square(pc_np[2, :])
    pc_np_range_square = pc_np_x_square + pc_np_z_square
    pc_mask_range = pc_np_range_square < opt.pc_max_range * opt.pc_max_range
    pc_np = pc_np[:, pc_mask_range]
    intensity_np = intensity_np[:, pc_mask_range]

    # remove the ground points!

    if pc_np.shape[1] > 2 * opt.input_pt_num:
        # point cloud too huge, voxel grid downsample first
        pc_np, intensity_np = downsample_with_reflectance(pc_np, intensity_np[0], voxel_grid_downsample_size=0.2)
        intensity_np = np.expand_dims(intensity_np, axis=0)
        pc_np = pc_np.astype(np.float32)
        intensity_np = intensity_np.astype(np.float32)
    # random sampling
    pc_np, intensity_np = downsample_np(pc_np, intensity_np, opt.input_pt_num)

    #  ------------- apply random transform on points under the NWU coordinate ------------
    Pr = np.identity(4, dtype=np.float32)
    Pr_inv = np.identity(4, dtype=np.float32)

    t_ij = P_cam_pc[0:3, 3]
    P = np.dot(P_cam_pc, Pr_inv)

    # now the point cloud is in CAMERA coordinate
    pc_homo_np = np.concatenate((pc_np, np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)), axis=0)  # 4xN
    Pr_pc_homo_np = np.dot(Pr, pc_homo_np)  # 4xN
    pc_np = Pr_pc_homo_np[0:3, :]  # 3xN

    # ------------ Farthest Point Sampling ------------------
    # node_a_np = fps_approximate(pc_np, voxel_size=4.0, node_num=self.opt.node_a_num)
    node_a_np, _ = farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                     int(opt.node_a_num * 8),
                                                                     replace=False)],
                                           k=opt.node_a_num)
    node_b_np, _ = farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                     int(opt.node_b_num * 8),
                                                                     replace=False)],
                                           k=opt.node_b_num)

    # -------------- convert to torch tensor ---------------------
    pc = torch.from_numpy(pc_np).to(device)  # 3xN
    intensity = torch.from_numpy(intensity_np).to(device)  # 1xN
    sn = torch.zeros(pc.size(), dtype=pc.dtype, device=pc.device).to(device)
    node_a = torch.from_numpy(node_a_np).to(device)  # 3xMa
    node_b = torch.from_numpy(node_b_np).to(device)  # 3xMb

    P = torch.from_numpy(P[0:3, :].astype(np.float32)).to(device)  # 3x4

    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).contiguous().to(device)  # 3xHxW
    K = torch.from_numpy(K.astype(np.float32)).to(device)  # 3x3

    t_ij = torch.from_numpy(t_ij.astype(np.float32)).to(device)  # 3

    return_value = pc, intensity, sn, node_a, node_b, P, img, K, t_ij
    
    # ------------ add batch size -----------------
    img = img.unsqueeze(0)
    pc = pc.unsqueeze(0)
    intensity = intensity.unsqueeze(0)
    sn = sn.unsqueeze(0)
    node_a = node_a.unsqueeze(0)
    node_b = node_b.unsqueeze(0)
    print(img.shape)
    B, H, W = img.size(0), img.size(2), img.size(3)
    N = pc.size(2)
    H_fine = int(round(H / opt.img_fine_resolution_scale))
    W_fine = int(round(W / opt.img_fine_resolution_scale))

    model.set_input(pc, intensity, sn, node_a, node_b,
                    P, img, K)

    coarse_prediction, fine_prediction = model.inference_pass()

    print("coardse_prediction: ", coarse_prediction)
    print(coarse_prediction.shape)
    # print(fine_prediction.shape)
