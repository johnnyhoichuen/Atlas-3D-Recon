import argparse
import shlex
import subprocess
import time
import yaml

from atlas.evaluation import eval_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare data + inference")
    parser.add_argument("--config", help="config file of the dataset")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(f'config: {config}')

    dataset = config['dataset']

    # inference
    img_dir = config['img_dir']
    pose_dir = config['pose_dir']

    # eval
    pred_path = f'results/{dataset}/{pose_dir}/{dataset}.ply'
    fine_tuned_params = config['fine_tuned_params'] # tx, ty, tz, scale
    gt_path_prefix = config['gt_path_prefix']
    gt_path = f'{gt_path_prefix}_{fine_tuned_params[:3]}_s{fine_tuned_params[3]}.ply'

    # print(f'img dir: {img_dir}')
    # print(f'pose dir: {pose_dir}')
    # print(f'pred path: {pred_path}')
    # print(f'gt path: {gt_path}')

    # # data selection
    # dataset, img_dir, pose_dir, gt_path = "ust_conf3_icp_opvs", "ust_conf_3", "opvs_s0.8", "data/ust_conf_iphone/gt/test_low_density_[2.89, 4.5, 0.5].ply" # todo: generate gt for ust_conf_3
    # # dataset, img_dir, pose_dir, gt_path = "ust_conf_iphone", "iphone", "direct", "data/ust_conf_iphone/gt/test_low_density_[2.89, 4.5, 0.5].ply" # gt with adjusted pose  # iphone with direct pose
    # # dataset, img_dir, pose_dir, gt_path = "ust_conf_iphone", "iphone", "from_quat", "data/ust_conf_iphone/gt/test_low_density_[2.89, 4.5, 0.5].ply" # gt with adjusted pose   # iphone with quat pose, (not yet tested)

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
                               f'--scenes meta/{dataset}/info.json '
                               f'--save_path results/{dataset}/{pose_dir}'))
    print(f'inference time used (voxel dim: [208, 208, 80]): {time.time() - start_time}s')

    # ###########################################################
    # ## evaluate
    # ###########################################################
    #
    # # # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_icp_opvs/opvs_s1.ply"  # 360 openvslam
    # # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_iphone/result-crop.ply"  # iphone
    # # gt_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/data/ust_conf_iphone/gt/iphone_atlas_ground_truth_1.pcd" # temporary gt
    #
    # # # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_icp_opvs/opvs_s1.ply"  # 360 openvslam
    # pred_path = f"results/{dataset}/{pose_dir}/{dataset}.ply"  # iphone
    # # gt_path = "data/ust_conf_iphone/gt/gt_1.ply" # temporary gt, unadjusted pose
    mesh_metrics = eval_mesh(file_pred=pred_path, file_trgt=gt_path)
    print(f'mesh metrics: {mesh_metrics}')

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
