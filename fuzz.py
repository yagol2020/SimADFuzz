import json
import math
import os
import shutil
import time

import joblib
import pandas as pd
import torch
from carla import Timestamp
from deap import base, tools, algorithms
from dataclasses import field

import random
import carla
import copyreg
import dill

import matplotlib.pyplot as plt
from shapely.geometry import LineString

from models import TransformerModel, sdc_feature_extract
import loguru


def location_pickle(location):
    return carla.Location, (location.x, location.y, location.z)


def rotation_pickle(rotation):
    return carla.Rotation, (rotation.pitch, rotation.yaw, rotation.roll)


def transform_pickle(transform):
    return carla.Transform, (
        carla.Location(transform.location.x, transform.location.y, transform.location.z),
        carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
    )


class MyFitness(base.Fitness):
    # model_fitness, sdc_fitness, min_distance, violation_num
    weights = (1, 1, -1, 1)


copyreg.pickle(carla.Location, location_pickle)
copyreg.pickle(carla.Rotation, rotation_pickle)
copyreg.pickle(carla.Transform, transform_pickle)

model = TransformerModel(car_num=3, car_features_size=7)
model.load_state_dict(torch.load("Transformer.ck.pth"))
model = model.cuda()
model.eval()
sdc = joblib.load("sdc.ck.pth")
ads_under_test = "interfuser"
violation_keeper = []


class Scenario:
    fitness: base.Fitness = field(default_factory=MyFitness)

    def __init__(
            self, town_str, result_path, npc_num, walker_num
    ):
        self.fitness = MyFitness()
        self.npc_num = npc_num
        self.walker_num = walker_num
        ##################################
        self.ego = {}
        self.npcs = []
        self.walker = []
        self.weather = {}
        self.town_str = town_str
        ##################################
        self.gid = 0
        self.cid = 0
        self.result_path = os.path.abspath(result_path)
        ##################################
        self.infos = {}

    def init_pick(self, pick_dir):
        assert os.path.exists(os.path.join(pick_dir, "ego.json"))
        assert os.path.exists(os.path.join(pick_dir, "npcs.json"))
        assert os.path.exists(os.path.join(pick_dir, "walkers.json"))
        assert os.path.exists(os.path.join(pick_dir, "weather.json"))
        self.ego = json.load(open(os.path.join(pick_dir, "ego.json")))
        self.npcs = json.load(open(os.path.join(pick_dir, "npcs.json")))
        self.walker = json.load(open(os.path.join(pick_dir, "walkers.json")))
        self.weather = json.load(open(os.path.join(pick_dir, "weather.json")))

    def ego_random(self):
        self.ego = {}
        sps = json.load(open(f"town_sps/{self.town_str}.sps.json", "r"))

        ego_sp_road = random.choice(list(sps.keys()))
        ego_sp = random.choice(sps[ego_sp_road])
        self.ego["x"] = ego_sp["location"]["x"]
        self.ego["y"] = ego_sp["location"]["y"]
        self.ego["z"] = ego_sp["location"]["z"]
        self.ego["pitch"] = ego_sp["rotation"]["pitch"]
        self.ego["yaw"] = ego_sp["rotation"]["yaw"]
        self.ego["roll"] = ego_sp["rotation"]["roll"]
        self.ego["road_id"] = ego_sp_road

        ego_dp_road = random.choice(list(sps.keys()))
        ego_dp = random.choice(sps[ego_dp_road])
        self.ego["x_dp"] = ego_dp["location"]["x"]
        self.ego["y_dp"] = ego_dp["location"]["y"]
        self.ego["z_dp"] = ego_dp["location"]["z"]
        self.ego["pitch_dp"] = ego_dp["rotation"]["pitch"]
        self.ego["yaw_dp"] = ego_dp["rotation"]["yaw"]
        self.ego["roll_dp"] = ego_dp["rotation"]["roll"]
        self.ego["road_id_dp"] = ego_dp_road

    def change_ego(self):
        sp_road = self.ego["road_id"]
        dp_road = self.ego["road_id_dp"]

        sps = json.load(open(f"town_sps/{self.town_str}.sps.json", "r"))
        new_sp = random.choice(sps[sp_road])
        new_dp = random.choice(sps[dp_road])

        self.ego["x"] = new_sp["location"]["x"]
        self.ego["y"] = new_sp["location"]["y"]
        self.ego["z"] = new_sp["location"]["z"]
        self.ego["yaw"] = new_sp["rotation"]["yaw"]
        self.ego["pitch"] = new_sp["rotation"]["pitch"]
        self.ego["roll"] = new_sp["rotation"]["roll"]

        self.ego["x_dp"] = new_dp["location"]["x"]
        self.ego["y_dp"] = new_dp["location"]["y"]
        self.ego["z_dp"] = new_dp["location"]["z"]
        self.ego["yaw_dp"] = new_dp["rotation"]["yaw"]
        self.ego["pitch_dp"] = new_dp["rotation"]["pitch"]
        self.ego["roll_dp"] = new_dp["rotation"]["roll"]

    def npc_random(self, generate_npc_num):
        """
        generate npc with random
        :param generate_npc_num: the number of npc you want to generate
        """
        assert len(self.npcs) + generate_npc_num == self.npc_num
        sps = json.load(open(f"town_sps/{self.town_str}.sps.json", "r"))
        for _ in range(generate_npc_num):
            npc = {}
            while True:
                is_generate_without_duplicate = True
                npc_sp_road = random.choice(list(sps.keys()))
                npc_sp = random.choice(sps[npc_sp_road])
                npc["x"] = npc_sp["location"]["x"]
                npc["y"] = npc_sp["location"]["y"]
                npc["z"] = npc_sp["location"]["z"]
                npc["pitch"] = npc_sp["rotation"]["pitch"]
                npc["yaw"] = npc_sp["rotation"]["yaw"]
                npc["roll"] = npc_sp["rotation"]["roll"]
                npc["road_id"] = npc_sp_road
                if npc["x"] == self.ego["x"] and npc["y"] == self.ego["y"]:
                    print("Duplicate npc with ego, regenerate")
                    is_generate_without_duplicate = False
                if is_generate_without_duplicate:
                    break
            npc_dp_road = random.choice(list(sps.keys()))
            npc_dp = random.choice(sps[npc_dp_road])
            npc["x_dp"] = npc_dp["location"]["x"]
            npc["y_dp"] = npc_dp["location"]["y"]
            npc["z_dp"] = npc_dp["location"]["z"]
            npc["pitch_dp"] = npc_dp["rotation"]["pitch"]
            npc["yaw_dp"] = npc_dp["rotation"]["yaw"]
            npc["roll_dp"] = npc_dp["rotation"]["roll"]
            npc["road_id_dp"] = npc_dp_road

            self.npcs.append(npc)
        assert len(self.npcs) == self.npc_num

    def walker_random(self, generate_walker_num):
        assert generate_walker_num + len(self.walker) == self.walker_num
        sps = dill.load(open(f"town_sps/{self.town_str}.sps.pick", "rb"))
        safe_distance = 10  # 定义与 ego 的安全距离
        for _ in range(generate_walker_num):
            target_ego = self.ego
            ego_sp_x, ego_sp_y = target_ego["x"], target_ego["y"]
            bound_x1, bound_x2, bound_y1, bound_y2 = ego_sp_x - 30, ego_sp_x + 30, ego_sp_y - 30, ego_sp_y + 30
            while True:  # 循环直到生成符合条件的坐标
                walker_sp_x = random.randint(int(bound_x1), int(bound_x2))
                walker_sp_y = random.randint(int(bound_y1), int(bound_y2))
                distance = math.sqrt((walker_sp_x - ego_sp_x) ** 2 + (walker_sp_y - ego_sp_y) ** 2)
                if distance >= safe_distance:
                    break  # 如果满足条件，退出循环
            walker_sp_x = random.randint(int(bound_x1), int(bound_x2))
            walker_sp_y = random.randint(int(bound_y1), int(bound_y2))
            walker_yaw = random.randint(0, 360)
            walker_speed = random.uniform(0, 10)
            r = random.randint(0, 100)
            if r <= 10:
                self.walker.append(
                    {"x": walker_sp_x, "y": walker_sp_y, "z": 1.5, "pitch": 0, "yaw": walker_yaw, "roll": 0,
                     "type": "IMMOBILE"})
            elif r <= 50:
                self.walker.append(
                    {"x": walker_sp_x, "y": walker_sp_y, "z": 1.5, "pitch": 0, "yaw": walker_yaw, "roll": 0,
                     "type": "LINEAR", "speed": walker_speed})
            else:
                walker_sp, walker_dp = random.choices(sps, k=2)
                self.walker.append(
                    {"x": walker_sp.location.x, "y": walker_sp.location.y, "z": walker_sp.location.z,
                     "pitch": walker_sp.rotation.pitch,
                     "yaw": walker_sp.rotation.yaw, "roll": walker_sp.rotation.roll,
                     "x_dp": walker_dp.location.x, "y_dp": walker_dp.location.y, "z_dp": walker_dp.location.z,
                     "pitch_dp": walker_dp.rotation.pitch, "yaw_dp": walker_dp.rotation.yaw,
                     "roll_dp": walker_dp.rotation.roll,
                     "type": "AUTOPILOT", "speed": walker_speed})
        assert len(self.walker) == self.walker_num

    def weather_random(self):
        self.weather["cloud"] = random.randint(0, 100)
        self.weather["rain"] = random.randint(0, 100)
        self.weather["puddle"] = random.randint(0, 100)
        self.weather["wind"] = random.randint(0, 100)
        self.weather["fog"] = random.randint(0, 100)
        self.weather["wetness"] = random.randint(0, 100)
        self.weather["angle"] = random.randint(0, 360)
        self.weather["altitude"] = random.randint(-90, 90)

    def change_npcs(self):
        """
        change npcs in the scenarios, make them stay in the road
        """
        for npc_index in range(len(self.npcs)):
            sp_road = self.npcs[npc_index]["road_id"]
            dp_road = self.npcs[npc_index]["road_id_dp"]

            sps = json.load(open(f"town_sps/{self.town_str}.sps.json", "r"))
            new_sp = random.choice(sps[sp_road])  # reselect one sp, stay in their original road
            new_dp = random.choice(sps[dp_road])

            self.npcs[npc_index]["x"] = new_sp["location"]["x"]
            self.npcs[npc_index]["y"] = new_sp["location"]["y"]
            self.npcs[npc_index]["z"] = new_sp["location"]["z"]
            self.npcs[npc_index]["yaw"] = new_sp["rotation"]["yaw"]
            self.npcs[npc_index]["pitch"] = new_sp["rotation"]["pitch"]
            self.npcs[npc_index]["roll"] = new_sp["rotation"]["roll"]

            self.npcs[npc_index]["x_dp"] = new_dp["location"]["x"]
            self.npcs[npc_index]["y_dp"] = new_dp["location"]["y"]
            self.npcs[npc_index]["z_dp"] = new_dp["location"]["z"]
            self.npcs[npc_index]["yaw_dp"] = new_dp["rotation"]["yaw"]
            self.npcs[npc_index]["pitch_dp"] = new_dp["rotation"]["pitch"]
            self.npcs[npc_index]["roll_dp"] = new_dp["rotation"]["roll"]

    def init_random(self):
        self.ego = {}
        self.npcs = []
        self.walker = []
        self.weather = {}
        #####################################################################################
        self.ego_random()
        #####################################################################################
        self.npc_random(generate_npc_num=self.npc_num)
        #####################################################################################
        self.walker_random(generate_walker_num=self.walker_num)
        #####################################################################################
        self.weather_random()
        #####################################################################################
        return self

    def check_walker(self):
        new_walker = []
        for walker in self.walker:
            walker_x = walker["x"]
            walker_y = walker["y"]
            distance = math.sqrt((walker_x - self.ego["x"]) ** 2 + (walker_y - self.ego["y"]) ** 2)
            if distance > 10:
                new_walker.append(walker)
        self.walker = new_walker
        self.walker_random(generate_walker_num=self.walker_num - len(self.walker))

    def dump_pick(self):
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")
        os.mkdir("temp_dir")
        os.mkdir("temp_dir/oracles")
        json.dump(self.ego, open("temp_dir/ego.json", "w"))
        json.dump(self.npcs, open("temp_dir/npcs.json", "w"))
        json.dump(self.walker, open("temp_dir/walkers.json", "w"))
        json.dump(self.weather, open("temp_dir/weather.json", "w"))
        json.dump(self.town_str, open("temp_dir/town_str.json", "w"))

    def check(self):
        assert self.infos == {}
        assert self.ego != {}
        assert len(self.npcs) == self.npc_num
        assert len(self.walker) == self.walker_num
        assert self.weather != {}


class Violation:
    def __init__(
            self,
            vio_type,
            event
    ) -> None:
        self.vio_type = vio_type
        if isinstance(event, tuple):
            self.vio_coords_x = event[1].location.x
            self.vio_coords_y = event[1].location.y
            self.vio_coords_z = event[1].location.z
            self.vio_time = event[0]
            if isinstance(self.vio_time, Timestamp):
                self.vio_time = self.vio_time.elapsed_seconds
        elif isinstance(event, dict):
            self.vio_coords_x = event["x"]
            self.vio_coords_y = event["y"]
            self.vio_coords_z = event["z"]
            self.vio_time = event["time_stamp"]

    @loguru.logger.catch()
    def equal(self, another_vio):
        if self.vio_type != another_vio.vio_type:
            return False
        if abs(self.vio_time - another_vio.vio_time) > 20:
            return False
        if (abs(self.vio_coords_x - another_vio.vio_coords_x) > 5 and
                abs(self.vio_coords_y - another_vio.vio_coords_y) > 5):
            return False
        return True


def eval_scenario(scenario):
    model_s = 0
    sdc_s = []
    min_dis = -99999
    vio_num = 0
    scenario.check()
    scenario.dump_pick()
    cmd = "ADS=interfuser DISPLAY_ENABLE=2 python sim_executor.py "
    cmd += "--ego-json temp_dir/ego.json "
    cmd += "--npcs-json temp_dir/npcs.json "
    cmd += "--walkers-json temp_dir/walkers.json "
    cmd += "--weather-json temp_dir/weather.json "
    cmd += f"--town {scenario.town_str} "
    cmd += "--save-path temp_dir/info.csv "
    cmd += "--carla-recorder-path temp_dir/recorder.rec "
    cmd += "--oracles-dir temp_dir/oracles "
    cmd += f"--ads {ads_under_test} "
    print(cmd)
    json.dump(time.time(), open("temp_dir/start_sim_time.json", "w"))
    os.system(cmd)
    json.dump(time.time(), open("temp_dir/end_sim_time.json", "w"))
    if not os.path.exists("temp_dir/oracles/state.pick"):
        print("Error in exec simulation, no result found")
        return model_s, sum(sdc_s), min_dis, vio_num
    try:
        scenario.infos = pd.read_csv("temp_dir/info.csv")
    except:
        return model_s, sum(sdc_s), min_dis, vio_num
    plt.figure(figsize=(8, 6))  # 创建一个新画布
    for car_id in range(1 + scenario.npc_num):
        ego_x = scenario.infos[[f"loc_x_{car_id}"]]
        ego_y = scenario.infos[[f"loc_y_{car_id}"]]
        plt.plot(ego_x, ego_y, label=f"V-{car_id}", marker="o", linestyle="-")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp_dir/trace.png", dpi=500)
    ################################################################################
    sdc_features = []
    ################################################################################
    df = scenario.infos.drop(columns=['dur_time'])
    loc_x_0_first = df['loc_x_0'].iloc[0]
    loc_y_0_first = df['loc_y_0'].iloc[0]
    for i in range(1 + scenario.npc_num):
        # 计算相对坐标
        df[f'loc_x_{i}'] -= loc_x_0_first
        df[f'loc_y_{i}'] -= loc_y_0_first
    df = df.drop_duplicates()
    for i in range(1 + scenario.npc_num):
        coord = df[[f"loc_x_{i}", f"loc_y_{i}"]].values
        sdc_features.append(sdc_feature_extract(coord, "sdc.json"))
    features = df.values
    features = torch.tensor(features, dtype=torch.float32).cuda()
    features = features.unsqueeze(0)
    ################################################################################
    model_s = model(features).cpu().detach().numpy()[0][0]  # violation prediction model
    ################################################################################
    for sdc_feature in sdc_features:
        sdc_s.append(sdc.predict_proba([sdc_feature])[0][1])  # sdc model
    ################################################################################
    state = dill.load(open("temp_dir/oracles/state.pick", "rb"))
    for event in state.collision_details:
        v = Violation("Collision", event=event)
        is_unique = True
        for v2 in violation_keeper:
            if v.equal(v2):
                is_unique = False
                break
        if is_unique:
            violation_keeper.append(v)
            vio_num += 1
    for event in state.speeding_details:
        v = Violation("Speeding", event=event)
        is_unique = True
        for v2 in violation_keeper:
            if v.equal(v2):
                is_unique = False
                break
        if is_unique:
            violation_keeper.append(v)
            vio_num += 1
    for event in state.laneinvasion_details:
        v = Violation("Lane Invasion", event=event)
        is_unique = True
        for v2 in violation_keeper:
            if v.equal(v2):
                is_unique = False
                break
        if is_unique:
            violation_keeper.append(v)
            vio_num += 1
    for event in state.stuck_details:
        v = Violation("Stuck", event=event)
        is_unique = True
        for v2 in violation_keeper:
            if v.equal(v2):
                is_unique = False
                break
        if is_unique:
            violation_keeper.append(v)
            vio_num += 1
    for event in state.running_red_light_details:
        v = Violation("Running Red Light", event=event)
        is_unique = True
        for v2 in violation_keeper:
            if v.equal(v2):
                is_unique = False
                break
        if is_unique:
            violation_keeper.append(v)
            vio_num += 1
    ################################################################################
    min_dis = int(json.load(open("temp_dir/oracles/min_dis.json", "r")))
    ################################################################################
    shutil.move("temp_dir", os.path.join(scenario.result_path, f"{scenario.gid}:{scenario.cid}"))
    print(f"Finished scenario: {scenario.gid}:{scenario.cid}")
    ################################################################################
    return model_s, sdc_s[0], min_dis, vio_num


def is_intersection(trajectory1, trajectory2):
    try:
        line1 = LineString(trajectory1)  # 将轨迹1转换为LineString
        line2 = LineString(trajectory2)  # 将轨迹2转换为LineString
    except:
        return False
    return line1.intersects(line2)  # 判断两条轨迹是否相交


def mate_scenario(scenario1, scenario2):
    assert scenario1.npc_num == scenario2.npc_num
    assert len(scenario1.walker) == len(scenario2.walker) == scenario1.walker_num == scenario2.walker_num
    #######################################################################################
    swap_pair = {
        "s1": [],
        "s2": []
    }
    #######################################################################################
    s1_ego_trace = scenario1.infos[[f"loc_x_0", f"loc_y_0"]].values
    s2_ego_trace = scenario2.infos[[f"loc_x_0", f"loc_y_0"]].values
    #######################################################################################
    for npc_id in range(scenario1.npc_num):
        coord = scenario1.infos[[f"loc_x_{npc_id + 1}", f"loc_y_{npc_id + 1}"]].values
        if is_intersection(s2_ego_trace, coord):
            swap_pair["s1"].append(npc_id)  # means that this list should be swap

    for npc_id in range(scenario2.npc_num):
        coord = scenario2.infos[[f"loc_x_{npc_id + 1}", f"loc_y_{npc_id + 1}"]].values
        if is_intersection(s1_ego_trace, coord):
            swap_pair["s2"].append(npc_id)

    if swap_pair["s1"] and swap_pair["s2"]:
        npc_id_s1 = random.choice(swap_pair["s1"])
        npc_id_s2 = random.choice(swap_pair["s2"])
        scenario1.npcs[npc_id_s1], scenario2.npcs[npc_id_s2] = scenario2.npcs[npc_id_s2], scenario1.npcs[
            npc_id_s1]
    elif swap_pair["s1"]:
        npc_id_s1 = random.choice(swap_pair["s1"])
        npc_id_s2 = random.randint(0, len(scenario2.npcs) - 1)
        scenario1.npcs[npc_id_s1], scenario2.npcs[npc_id_s2] = scenario2.npcs[npc_id_s2], scenario1.npcs[
            npc_id_s1]
    elif swap_pair["s2"]:
        npc_id_s2 = random.choice(swap_pair["s2"])
        npc_id_s1 = random.randint(0, len(scenario1.npcs) - 1)
        scenario1.npcs[npc_id_s1], scenario2.npcs[npc_id_s2] = scenario2.npcs[npc_id_s2], scenario1.npcs[
            npc_id_s1]
    #######################################################################################
    swap_walker_idx = random.choices(range(scenario1.walker_num), k=scenario1.walker_num // 2)
    for idx in swap_walker_idx:
        # 交换相同位置的 walkers
        scenario1.walker[idx], scenario2.walker[idx] = scenario2.walker[idx], scenario1.walker[idx]
    #######################################################################################
    assert scenario1.npc_num == scenario2.npc_num
    assert len(scenario1.walker) == len(scenario2.walker) == scenario1.walker_num == scenario2.walker_num
    #######################################################################################
    return scenario1, scenario2


def find_vehicles_not_closing_in(infos, npc_num, pat):
    def euclidean_distance(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    times = infos['dur_time'].values
    vehicles_not_closing_in = set()
    ego_x = infos[f"loc_x_0"].values
    ego_y = infos[f"loc_y_0"].values

    for npc_idx in range(npc_num):
        x_a = infos[f'loc_x_{npc_idx + 1}'].values
        y_a = infos[f'loc_y_{npc_idx + 1}'].values
        is_not_closing = True

        for start_time in range(len(times) - pat + 1):
            distances = [
                euclidean_distance(ego_x[start_time + i], ego_y[start_time + i], x_a[start_time + i],
                                   y_a[start_time + i])
                for i in range(pat)
            ]
            deltas = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
            if sum(deltas) < 0:
                print(f"The NPC {npc_idx} is closing to ego in time {start_time} to {start_time + pat}")
                is_not_closing = False
                break

        if is_not_closing:
            print(f"NPC {npc_idx} is not closing to ego, remove it")
            vehicles_not_closing_in.add(npc_idx)
        else:
            print(f"NPC {npc_idx} sometime is closing to ego, keep it")

    return vehicles_not_closing_in


def get_route_distance(coords):
    total_length = 0
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total_length


def mutate_scenario(scenario):
    ##################################################
    # Check is ego stuck too long
    ##################################################
    ego_route_dis = get_route_distance(scenario.infos[["loc_x_0", "loc_y_0"]].values)
    if ego_route_dis < 10 or random.random() < 0.5:  # the route of ego is too small, maybe stuck, reselect one route for ego
        return (scenario.init_random(),)
    ##################################################
    scenario.change_ego()  # mutate ego keep original road
    ##################################################
    be_delete = set()
    ##################################################
    vehicles = find_vehicles_not_closing_in(infos=scenario.infos, pat=60, npc_num=scenario.npc_num)
    for v_id in vehicles:
        print(f"Delete {v_id} by not closing")
        be_delete.add(v_id)
    ##################################################
    for npc_id in range(scenario.npc_num):
        trace_dis = get_route_distance(scenario.infos[[f"loc_x_{npc_id + 1}", f"loc_y_{npc_id + 1}"]].values)
        if trace_dis < 5:
            print(f"Delete {npc_id} by do not move")
            be_delete.add(npc_id)
    scenario.npcs = [npc for npc_id, npc in enumerate(scenario.npcs) if npc_id not in be_delete]
    ##################################################
    scenario.change_npcs()  # mutate remain npc, make them stay in original road
    scenario.npc_random(generate_npc_num=scenario.npc_num - len(scenario.npcs))
    ##################################################
    scenario.walker = []
    scenario.walker_random(generate_walker_num=scenario.walker_num)
    ##################################################
    scenario.weather_random()
    ##################################################
    return (scenario,)


def load_seed_scenarios_random(res_path, pop_size, npc_num, town_str, walker_num):
    ret = []
    for i in range(pop_size):
        s = Scenario(town_str=town_str, result_path=res_path, npc_num=npc_num, walker_num=walker_num)
        s.init_random()
        ret.append(s)
    return ret


def load_seed_scenarios(seed_dir, res_path, pop_size, npc_num, town_str, walker_num):
    ret = []
    for seed in os.listdir(seed_dir):
        pth = os.path.join(seed_dir, seed)
        s = Scenario(town_str=town_str, result_path=res_path, npc_num=npc_num, walker_num=walker_num)
        s.init_pick(pth)
        ret.append(s)
    if len(ret) == pop_size:
        print(f"Find {pop_size} seeds, return this")
    elif len(ret) < pop_size:
        print(f"Try to generate {pop_size - len(ret)} seeds")
        ret2 = load_seed_scenarios_random(res_path=res_path, pop_size=pop_size - len(ret), npc_num=npc_num,
                                          town_str=town_str, walker_num=walker_num)
        ret.extend(ret2)
    else:
        print(f"Too many seeds {len(ret)}, get first {pop_size}")
        ret = ret[:pop_size]
    return ret


if __name__ == "__main__":
    pop_size = 10
    npc_num = 2
    walker_num = 10
    town_str = "Town03"
    fuzzing_start_time = time.time()
    res_path = f"result/{fuzzing_start_time}-ads:{ads_under_test}"
    os.mkdir(res_path)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", eval_scenario)
    toolbox.register("mate", mate_scenario)
    toolbox.register("mutate", mutate_scenario)
    toolbox.register("select", tools.selNSGA2)
    # 初始种子
    initial_seed_scenarios = load_seed_scenarios("seeds", res_path=res_path, pop_size=pop_size,
                                                 npc_num=npc_num, town_str=town_str, walker_num=walker_num)
    for i, indv in enumerate(initial_seed_scenarios):
        indv.gid = 0
        indv.cid = i
        indv.infos = {}

    valid_individuals = []
    all_fitness = toolbox.map(toolbox.evaluate, initial_seed_scenarios)
    for ind, fit in zip(initial_seed_scenarios, all_fitness):
        ind.fitness.values = fit
        if fit[2] == -99999:
            continue
        valid_individuals.append(ind)

    scenarios = valid_individuals
    cur_gen = 1
    while time.time() - fuzzing_start_time < 6 * 60 * 60:
        offsprings = algorithms.varOr(
            scenarios, toolbox, lambda_=pop_size, cxpb=0.5, mutpb=0.5
        )
        for index, c in enumerate(offsprings):
            c.gid = cur_gen
            c.cid = index
            c.infos = {}
        valid_individuals = []
        all_fitness = toolbox.map(toolbox.evaluate, offsprings)
        for ind, fit in zip(offsprings, all_fitness):
            ind.fitness.values = fit
            if fit[2] == -99999:
                continue
            valid_individuals.append(ind)

        scenarios = toolbox.select(scenarios + valid_individuals, pop_size)
        cur_gen += 1
