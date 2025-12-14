#!/bin/bash

# Download dataset
python scripts/download_hf_dataset.py --path ./dataset


# Playback each task
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_box_cleanup.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_can_sort_random.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_coffee.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_drawer_cleanup.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_lift_tray.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_pouring.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_threading.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_three_piece_assembly.hdf5 --n 100 --video_skip 1
python scripts/playback_datasets.py --dataset ./dataset/generated/two_arm_transport.hdf5 --n 100 --video_skip 1