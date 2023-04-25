import os

import numpy as np
import oxford.options
import torch

from models.multimodal_classifier import MMClassifer, MMClassiferCoarse

if __name__ == "__main__":
    opt = oxford.options.Options()

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
