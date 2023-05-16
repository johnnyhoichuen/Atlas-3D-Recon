import argparse
import math
import os.path
import open3d as o3d
import yaml

from atlas.evaluation import eval_mesh

def manual_adjust(config):
    dataset = config['dataset']
    pose_dir = config['pose_dir']
    pred_path = f'results/{dataset}/{pose_dir}/{dataset}.ply'
    gt_path_prefix = config['gt_path_prefix']

    manual_rotate = config['manual_rotate']
    manual_translate = config['manual_translate']
    manual_scale = config['manual_scale']

    pcd = o3d.io.read_point_cloud(f'{gt_path_prefix}.ply')

    # rotate
    rads = [math.radians(angle) for angle in manual_rotate] # x, z, y in meshlab, ust_conf_3
    R = pcd.get_rotation_matrix_from_xyz(rads)
    pcd = pcd.rotate(R, center=(0, 0, 0))  # center = rotation center

    # translate
    test_pcd = o3d.geometry.PointCloud(pcd).translate(manual_translate)

    # scale
    test_pcd.scale(manual_scale, center=test_pcd.get_center())

    # write if needed
    updated_gt_path = f'{gt_path_prefix}_{manual_translate}_s{manual_scale}.ply'
    if not os.path.exists(updated_gt_path):
        o3d.io.write_point_cloud(updated_gt_path, test_pcd)

    # evaluate
    json_metrics = eval_mesh(file_pred=pred_path, file_trgt=updated_gt_path)
    json_metrics["translation"] = manual_translate
    json_metrics["scale"] = manual_scale

    print(f'mesh metrics from translation: {manual_translate}, scale: {manual_scale}: \n{json_metrics}\n')

def fine_tune(config):
    dataset = config['dataset']
    pose_dir = config['pose_dir']
    pred_path = f'results/{dataset}/{pose_dir}/{dataset}.ply'
    gt_path_prefix = config['gt_path_prefix']
    original_gt_path = f'{gt_path_prefix}.ply'

    manual_rotate = config['manual_rotate'] # for now, we just eye-ball the right rotation in meshlab
    x_range = config['x_range']
    y_range = config['y_range']
    z_range = config['z_range']
    scales = config['scales']

    pcd = o3d.io.read_point_cloud(original_gt_path)

    # rotation
    rads = [math.radians(angle) for angle in manual_rotate]  # x, z, y in meshlab
    R = pcd.get_rotation_matrix_from_xyz(rads)
    pcd = pcd.rotate(R, center=(0, 0, 0))  # center = rotation center

    eval_results = []
    counter = 1
    total = len(x_range) * len(y_range) * len(z_range) * len(scales)

    print(f'evaluating prediction: {pred_path}')

    # iterate all possibilities to find the optimal f-score
    for x in x_range:
        for y in y_range:
            for z in z_range:
                for scale in scales:
                    # translation
                    translation = [x, y, z]
                    test_pcd = o3d.geometry.PointCloud(pcd).translate(translation) # needa deep copy else the only pcd will be manipulated many times
                    # mesh_s = copy.deepcopy(mesh).translate((2, 0, 0))

                    # scale
                    test_pcd.scale(scale, center=test_pcd.get_center())

                    # write if needed
                    updated_gt_path = f'{gt_path_prefix}_{translation}_s{scale}.ply'
                    if not os.path.exists(updated_gt_path):
                        o3d.io.write_point_cloud(updated_gt_path, test_pcd)

                    # evaluate
                    json_metrics = eval_mesh(file_pred=pred_path, file_trgt=updated_gt_path)
                    json_metrics["translation"] = translation
                    eval_results.append(json_metrics)

                    print(f'{counter}/{total} mesh metrics from translation: {translation}, scale: {scale}: \n{json_metrics}\n')
                    counter += 1

    # find the best metrics according to fscore
    sorted_metrics = sorted(eval_results, key=lambda x: x['fscore'])
    for met in sorted_metrics:
        fscore = met['fscore']
        translation = met['translation']
        print(f'{translation} fscore: {fscore}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(f'config: {config}')

    # # manual scaling
    # manual_adjust(config=config)

    # fine tuning
    fine_tune(config=config)