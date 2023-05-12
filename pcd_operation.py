import argparse
import json
import math

import open3d as o3d
import numpy as np

from atlas.evaluation import eval_mesh

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--pcd_path", required=True)
    # args = parser.parse_args()

    pcd_path = 'data/ust_conf_iphone/gt/gt_1_low_density.ply'

    pcd = o3d.io.read_point_cloud(pcd_path)
    rads = [math.radians(angle) for angle in [90, -90, 0]] # x, z, y in meshlab

    R = pcd.get_rotation_matrix_from_xyz(rads)
    pcd = pcd.rotate(R, center=(0, 0, 0))  # center = rotation center

    # find a rough number on meshlab, then fine tune with program
    x_possible = [2.85, 2.90, 2.95]
    y_possible = [4.40, 4.45, 4.50]
    z_possible = [0.65, 0.70, 0.75, 0.8]

    # for debugging
    # x_possible, y_possible, z_possible = [2.85], [4.40], [0.60]

    result = []
    counter = 1
    total = len(x_possible) * len(y_possible) * len(z_possible)

    # iterate all possibilities to find the optimal f-score
    for x in x_possible:
        for y in y_possible:
            for z in z_possible:
                # trans
                translation = [x, y, z]
                test_pcd = o3d.geometry.PointCloud(pcd).translate(translation)  # copy else only one pcd will be manipulated
                o3d.io.write_point_cloud(f'data/ust_conf_iphone/gt/test_low_density_{translation}.ply', test_pcd)

                # evaluate
                pred_path = f"results/ust_conf_iphone/direct/ust_conf_iphone.ply"  # iphone
                gt_path = f"data/ust_conf_iphone/gt/test_low_density_{translation}.ply"  # gt with adjusted pose
                json_metrics = eval_mesh(file_pred=pred_path, file_trgt=gt_path)
                json_metrics["translation"] = translation
                result.append(json_metrics)
                print(f'{counter}/{total} mesh metrics from {translation}: \n{json_metrics}\n')

                counter += 1

    # find the best metrics according to fscore
    sorted_metrics = sorted(result, key=lambda x: x['fscore'])
    for met in sorted_metrics:
        fscore = met['fscore']
        translation = met['translation']
        print(f'{translation} fscore: {fscore}')
