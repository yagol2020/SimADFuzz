import shutil

import os
import json
import subprocess
import sys
import time

import carla
import pandas as pd
from matplotlib import pyplot as plt

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--carla-recorder", default=False, action="store_true")
arg_parser.add_argument("--carla-recorder-path", type=str)

arg_parser.add_argument("--simadfuzz-recorder", default=False, action="store_true")
arg_parser.add_argument("--simadfuzz-recorder-dir", type=str)

args = arg_parser.parse_args()
recorder_type = "ERROR"
recorder_type = "CARLA" if args.carla_recorder else recorder_type
recorder_type = "SIMADFUZZ" if args.simadfuzz_recorder else recorder_type

if recorder_type == "CARLA":
    subprocess.Popen(["./run_carla.sh", "13000", "True"])
    time.sleep(10)
    client = carla.Client("localhost", 13000)
    client.set_timeout(10)
    client.set_replayer_time_factor(2.0)
    shutil.copy(args.carla_recorder_path, "/tmp/carla.rec")
    os.system(
        f"docker cp /tmp/carla.rec carla-sim-13000:/tmp/carla.rec")
    print(client.replay_file("/tmp/carla.rec", 0, 0, 0))
elif recorder_type == "SIMADFUZZ":
    shutil.rmtree("temp_dir")
    shutil.copytree(args.simadfuzz_recorder_dir, "temp_dir/")
    cmd = "ADS=interfuser python sim_executor.py --debug --open-carla-display --track-ego"
    print(cmd)
    os.system(cmd)
    plt.figure(figsize=(8, 6))  # 创建一个新画布
    infos = pd.read_csv("temp_dir/info.csv")
    for car_id in range(3):
        ego_x = infos[[f"loc_x_{car_id}"]]
        ego_y = infos[[f"loc_y_{car_id}"]]
        plt.plot(ego_x, ego_y, label=f"V-{car_id}", marker="o", linestyle="-")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp_dir/trace.png", dpi=500)
else:
    print("Error in recorder type")
    sys.exit(-1)
