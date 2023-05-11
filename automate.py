import argparse
import shlex
import subprocess

from atlas.evaluation import eval_mesh

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="prepare data + inference")
    # # parser.add_argument("")
    # args = parser.parse_args()

    ###########################################################
    ## prepare data
    ###########################################################
    # dataset, color, pose_dir = "ust_conf3_icp_opvs", "ust_conf_3", "opvs_original"
    dataset, img_dir, pose_dir = "ust_conf_iphone", "iphone", "direct"  # iphone with direct pose
    # dataset, img_dir, pose_dir = "ust_conf_iphone", "iphone", "from_quat"  # iphone with quat pose
    subprocess.run(shlex.split(f"python3 prepare_data.py --path data --path_meta meta --dataset {dataset} "
                               f"--img_dir {img_dir} --pose_dir {pose_dir}"))

    ###########################################################
    ## inference
    ###########################################################
    subprocess.run(shlex.split(f'python3 inference.py --model results/release/semseg/final.ckpt '
                               f'--scenes meta/{dataset}/info.json '
                               f'--save_path results/{dataset}/{pose_dir}'))

    ###########################################################
    ## evaluate
    ###########################################################

    # # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_icp_opvs/opvs_s1.ply"  # 360 openvslam
    # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_iphone/result-crop.ply"  # iphone
    # gt_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/data/ust_conf_iphone/gt/iphone_atlas_ground_truth_1.pcd" # temporary gt

    # # pred_path = "/home/guest1/Documents/johnny/3d-recon/Atlas/results/ust_conf3_icp_opvs/opvs_s1.ply"  # 360 openvslam
    pred_path = f"results/ust_conf_iphone/{pose_dir}/{dataset}.ply"  # iphone
    gt_path = "data/ust_conf_iphone/gt/gt_1.ply" # temporary gt
    mesh_metrics = eval_mesh(file_pred=pred_path, file_trgt=gt_path)
    print(f'mesh metrics: {mesh_metrics}')

    ###################################

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
