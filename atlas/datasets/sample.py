# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import json
import os

import numpy as np


def prepare_sample_scene(img_dir, pose_dir, dataset, dataset_path, path_meta, verbose=2):
    """Generates a json file for a sample scene in our common format

    This wraps all the data about a scene into a single common format
    that is readable by our dataset class. This includes paths to all
    the images, depths, etc, as well as other metadata like camera
    intrinsics and pose. It also includes scene level information like
    a path to the mesh.

    Args:
        dataset: name of the scene.
            examples: 'scans/scene0000_00'
                      'scans_test/scene0708_00'
        dataset_path: path to the original data
        path_meta: path to where the generated data is saved.
            This can be the same as path, but it is recommended to
            keep them seperate so the original data is not accidentally
            modified. The generated data is saved into a mirror directory
            structure.

    Output:
        Creates the file path_meta/scene/info.json
        
    JSON format:
        {'dataset': 'sample',
         'path': path,
         'scene': scene,
         'frames': [{'file_name_image': '',
                     'intrinsics': intrinsics,
                     'pose': pose,
                     }
                   ]
         }

    """

    if verbose>0:
        print('preparing %s' % dataset)

    data = {'dataset': 'sample',
            'path': dataset_path,
            'scene': dataset,
            'frames': []
           }

    # load intrinsics for a specific dataset (except for iphone which has to load frame by frame)
    if dataset != "ust_conf_iphone":
        print('loading dataset-wide intrinsics')
        intrinsics = np.loadtxt(os.path.join(dataset_path, 'intrinsics.txt'))  # intrinsics are unified under the /data folder

    frame_ids = os.listdir(os.path.join(dataset_path, f'color/{img_dir}'))
    frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
    frame_ids = sorted(frame_ids)

    for i, frame_id in enumerate(frame_ids):
        if verbose>1 and i%25==0:
            print('preparing %s frame %d/%d' % (dataset, i, len(frame_ids)))

        if dataset == "ust_conf_iphone":
            # load intrinsics for iphone (diff intrinsics for each frame, but only applicable to iphone)
            intrinsics = np.loadtxt(os.path.join(dataset_path, "intrinsics" , '%05d.txt' % frame_id))

            # load pose & image
            pose = np.loadtxt(os.path.join(dataset_path, f'pose/{pose_dir}', '%05d.txt' % frame_id))
            # pose[0, 3] -= 2
            # pose[1, 3] -= 1  # todo: verify this: modify z-axis (which is y-axis in meshlab or after backprojection)

            img = os.path.join(dataset_path, f'color/{img_dir}', '%05d.jpg' % frame_id)
        else:
            # load pose & image
            pose = np.loadtxt(os.path.join(dataset_path, f'pose/{pose_dir}', '%08d.txt' % frame_id))
            img = os.path.join(dataset_path, f'color/{img_dir}', '%08d.jpg' % frame_id)

        # skip frames with no valid pose
        if not np.all(np.isfinite(pose)):
            continue

        frame = {'file_name_image': img,
                 'intrinsics': intrinsics.tolist(),
                 'pose': pose.tolist(),
                 }
        data['frames'].append(frame)

    os.makedirs(os.path.join(path_meta), exist_ok=True)
    json.dump(data, open(os.path.join(path_meta, 'info.json'), 'w'))



