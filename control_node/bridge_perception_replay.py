#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Replay recorded depth frames through bridge_perception offline.

用途：G1 门槛的离线验证 —— 把样台/仿真录制的深度帧（.npy，单位米）
喂给观测器，输出每帧 JSONL 结果，供与外部量测的真值对比。

用法（安装后）::

    bridge_perception_replay <depth_dir> --out obs.jsonl \\
        --width 640 --height 480 --hfov 1.22 --roll 0.0 --pitch 0.45

rosbag2 的提取（深度 topic -> .npy 序列）在容器里用 ros2 bag + 小脚本
完成；本工具刻意不依赖 ROS，可在任何有 numpy 的机器上运行。
"""

import argparse
import glob
import json
import os

import numpy as np

from control_node.bridge_perception import (
    BridgePerceptionConfig,
    CameraIntrinsics,
    bridge_observation,
)


def main(argv=None):
    """Parse arguments and write one bridge observation per depth frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_dir', help='directory containing *.npy depth frames (meters)')
    parser.add_argument('--out', default='bridge_observations.jsonl')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--hfov', type=float, default=1.22,
                        help='horizontal FOV in rad (sim baseline; use calibrated value)')
    parser.add_argument('--roll', type=float, default=0.0, help='camera roll (rad)')
    parser.add_argument('--pitch', type=float, default=0.0, help='camera pitch down (rad)')
    args = parser.parse_args(argv)

    intr = CameraIntrinsics.from_horizontal_fov(args.width, args.height, args.hfov)
    config = BridgePerceptionConfig()

    frames = sorted(glob.glob(os.path.join(args.depth_dir, '*.npy')))
    if not frames:
        raise SystemExit(f'no *.npy depth frames found in {args.depth_dir}')

    n_valid = 0
    with open(args.out, 'w', encoding='utf-8') as fp:
        for path in frames:
            depth = np.load(path)
            obs = bridge_observation(
                depth, intr,
                camera_roll=args.roll, camera_pitch=args.pitch, config=config,
            )
            obs['frame'] = os.path.basename(path)
            fp.write(json.dumps(obs, ensure_ascii=False) + '\n')
            if obs.get('valid'):
                n_valid += 1

    print(f'{len(frames)} frames processed, {n_valid} valid -> {args.out}')


if __name__ == '__main__':
    main()
