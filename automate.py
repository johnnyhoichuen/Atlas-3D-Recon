import argparse
import csv
import os.path
import shlex
import subprocess
import time
from pathlib import Path

import yaml

from atlas.evaluation import eval_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare data + inference")
    parser.add_argument("--config", help="config file of the dataset")
    parser.add_argument("--num_frames_inference", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(f'config: {config}')

    # prep data & inference input
    dataset = config['dataset']
    img_dir = config['img_dir']
    pose_dir = config['pose_dir']

    # number of frames used in inference.py
    num_frames_inference = args.num_frames_inference

    # eval input
    if num_frames_inference == -1:
        pred_path = f'results/{dataset}/{pose_dir}/{dataset}.ply'
    else:
        pred_path = f'results/{dataset}/{pose_dir}_{num_frames_inference}/{dataset}_{num_frames_inference}.ply'
    fine_tuned_params = config['fine_tuned_params'] # tx, ty, tz, scale
    gt_path_prefix = config['gt_path_prefix']
    gt_density = config['gt_density']
    gt_path = f'{gt_path_prefix}_{gt_density}_{fine_tuned_params[:3]}_s{fine_tuned_params[3]}.ply'

    ###########################################################
    ## prepare data
    ###########################################################
    subprocess.run(shlex.split(f"python3 prepare_data.py --path data --path_meta meta --dataset {dataset} "
                               f"--img_dir {img_dir} --pose_dir {pose_dir}"))

    ###########################################################
    ## inference
    ###########################################################
    start_time = time.time()
    subprocess.run(shlex.split(f'python3 inference.py --model results/release/semseg/final.ckpt '
                               f'--scenes meta/{dataset}/info.json '  # rmb to add 1 extra at the end of this 
                               f'--num_frames {num_frames_inference} '  # error: once not -1, it will not work
                               f'--save_path results/{dataset}/{pose_dir} '
                               ))

    inference_time = time.time() - start_time
    print(f'inference time used (voxel dim: [208, 208, 80]): {inference_time}s')

    ###########################################################
    ## evaluate
    ###########################################################
    mesh_metrics = eval_mesh(file_pred=pred_path, file_trgt=gt_path)
    print(f'mesh metrics: {mesh_metrics}')

    # save to csv
    # csv_path = Path(f"results/{dataset}/metrics.csv")
    csv_path = f"results/{dataset}/metrics.csv"
    with open(csv_path, 'a') as f:
        print(f'using csv')
        writer = csv.writer(f)

        # write header if csv doesn't exist
        if f.tell() == 0:
            writer.writerow(["num_frames", "time", "fscore", "dist1", "dist2", "prec", "recal"])

        # write data
        writer.writerow([args.num_frames_inference, inference_time, mesh_metrics['fscore'], mesh_metrics['dist1'], mesh_metrics['dist2'], mesh_metrics['prec'], mesh_metrics['recal']])


    # ###################################

    # openvslam
    # pose = np.loadtxt(os.path.join(path, dataset, 'ust_conf3_icp_opvs/opvs_original_pose', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s2_ry180_pose', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s2_pose', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.9_pose', '%08d.txt' % frame_id)) # starts to have some artifacts
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.8_pose', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.75_pose', '%08d.txt' % frame_id)) # best
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.6_pose', '%08d.txt' % frame_id)) # things start to fuse together & weird floor
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.5_pose', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.75_ry180_pose', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/opvs_s0.75_ry180_pose', '%08d.txt' % frame_id)) # aligning to icp's rotation

    # icp
    # pose = np.loadtxt(os.path.join(path, scene, 'ust_conf3_icp_opvs/icp_s0.375_pose', '%08d.txt' % frame_id)) # failed
    # pose = np.loadtxt(os.path.join(path, scene, 'pose', '%d.txt' % frame_id))

    # babar_skip_2_4x
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_opvs_skip_2_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_opvs_skip_2_s0.4_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_opvs_skip_2_s0.6_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_opvs_skip_2_s0.8_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_opvs_skip_2_s1.2_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation

    # babar_robot
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_robot_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_robot_s0.6_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_robot_s0.8_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_robot_s1.2_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation
    # pose = np.loadtxt(os.path.join(path, scene, 'babar_robot/pose_babar_robot_s1.4_4x', '%08d.txt' % frame_id)) # aligning to icp's rotation

    # fov test
    # pose = np.loadtxt(os.path.join(path, scene, f'{scene}/poses_{fov}fov', '%08d.txt' % frame_id))
    # pose = np.loadtxt(os.path.join(path, scene, f'{scene}/poses-90fov-original', '%08d.txt' % frame_id))
