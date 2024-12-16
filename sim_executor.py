import json
import math
import random
import shutil
import sys
from opcode import opname
from types import SimpleNamespace

import carla
import argparse
import copyreg
import dill
import os
import subprocess
import time

import pandas as pd

from agents.navigation.behavior_agent import BehaviorAgent

if os.environ["ADS"] == "interfuser":
    from InterFuser.team_code.interfuser_agent import InterfuserAgent
    from InterFuser.leaderboard.utils.route_manipulation import (
        interpolate_trajectory as interpolate_trajectory_interfuser,
    )
    from InterFuser.leaderboard.autoagents.agent_wrapper import (
        AgentWrapper as AgentWrapper_interfuser,
    )
    from InterFuser.srunner.scenariomanager.timer import (
        GameTime as GameTime_interfuser,
    )
    from InterFuser.srunner.scenariomanager.carla_data_provider import (
        CarlaDataProvider as CarlaDataProvider_interfuser,
    )


def location_pickle(location):
    return carla.Location, (location.x, location.y, location.z)


def rotation_pickle(rotation):
    return carla.Rotation, (rotation.pitch, rotation.yaw, rotation.roll)


def transform_pickle(transform):
    return carla.Transform, (
        carla.Location(transform.location.x, transform.location.y, transform.location.z),
        carla.Rotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
    )


def timestamp_pickle(timestamp):
    return carla.Timestamp, (
        timestamp.frame, timestamp.elapsed_seconds, timestamp.delta_seconds, timestamp.platform_timestamp
    )


def transform_2_location(transform):
    return carla.Location(
        x=transform.location.x, y=transform.location.y, z=transform.location.z
    )


def json_2_transform(infos, only_sp=False):
    sp = carla.Transform(
        carla.Location(x=infos["x"], y=infos["y"], z=infos["z"]),
        carla.Rotation(pitch=infos["pitch"], yaw=infos["yaw"], roll=infos["roll"])
    )
    if only_sp:
        return sp
    dp = carla.Transform(
        carla.Location(x=infos["x_dp"], y=infos["y_dp"], z=infos["z_dp"]),
        carla.Rotation(pitch=infos["pitch_dp"], yaw=infos["yaw_dp"], roll=infos["roll_dp"])
    )
    return sp, dp


def _on_invasion(event, state):
    if state.early_stop:
        return
    crossed_lanes = event.crossed_lane_markings
    for crossed_lane in crossed_lanes:
        if crossed_lane.lane_change == carla.LaneChange.NONE and crossed_lane.type != carla.LaneMarkingType.NONE:
            state.laneinvaded = True
            state.laneinvasion_details.append((event.timestamp, event.transform))
            state.violation_found = True


def _on_collision(event, state):
    if state.early_stop:
        return
    if event.other_actor.type_id != "static.road":
        state.crashed = True
        state.collision_details.append((event.timestamp, event.transform))
        state.early_stop = True
        state.early_stop_reason = "Ego collision"
        print(f"Collision! with {event.other_actor}", event)
        state.violation_found = True


def _on_front_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata/front-{image.frame}.jpg")


def _on_top_camera_capture(image):
    image.save_to_disk(f"/tmp/fuzzerdata/top-{image.frame}.jpg")


def save_video(video_dir):
    print("Saving front camera video", end=" ")
    vid_filename = f"{video_dir}/front.mp4"
    if os.path.exists(vid_filename):
        os.remove(vid_filename)
    cmd_cat = "ls /tmp/fuzzerdata/front-*.jpg | sort -V | xargs -I {} cat {}"
    cmd_ffmpeg = " ".join([
        "ffmpeg",
        "-f image2pipe",
        f"-r 20",
        "-vcodec mjpeg",
        "-i -",
        "-vcodec libx264",
        vid_filename
    ])

    cmd = f"{cmd_cat} | {cmd_ffmpeg} 2> /dev/null"
    os.system(cmd)
    print("(done)")

    cmd = "rm -f /tmp/fuzzerdata/front-*.jpg"
    os.system(cmd)

    print("Saving top camera video", end=" ")

    vid_filename = f"{video_dir}/top.mp4"
    if os.path.exists(vid_filename):
        os.remove(vid_filename)

    cmd_cat = "ls /tmp/fuzzerdata/top-*.jpg | sort -V | xargs -I {} cat {}"
    cmd_ffmpeg = " ".join([
        "ffmpeg",
        "-f image2pipe",
        f"-r 20",
        "-vcodec mjpeg",
        "-i -",
        "-vcodec libx264",
        vid_filename
    ])

    cmd = f"{cmd_cat} | {cmd_ffmpeg} 2> /dev/null"
    os.system(cmd)
    print("(done)")

    cmd = "rm -f /tmp/fuzzerdata/top-*.jpg"
    os.system(cmd)


if __name__ == "__main__":
    copyreg.pickle(carla.Location, location_pickle)
    copyreg.pickle(carla.Rotation, rotation_pickle)
    copyreg.pickle(carla.Transform, transform_pickle)
    copyreg.pickle(carla.Timestamp, timestamp_pickle)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--ego-json", type=str, default="temp_dir/ego.json")
    arg_parser.add_argument("--npcs-json", type=str, default="temp_dir/npcs.json")
    arg_parser.add_argument("--walkers-json", type=str, default="temp_dir/walkers.json")
    arg_parser.add_argument("--weather-json", type=str, default="temp_dir/weather.json")
    arg_parser.add_argument("--town", type=str, default="Town03")
    arg_parser.add_argument("--save-path", type=str, default="temp_dir/info.csv")
    arg_parser.add_argument("--carla-recorder-path", type=str, default="temp_dir/recorder.rec")
    arg_parser.add_argument("--oracles-dir", type=str, default="temp_dir/oracles")
    arg_parser.add_argument("--ads", type=str, default="interfuser")
    ################################################################################################################
    arg_parser.add_argument("--debug", default=False, action="store_true")
    arg_parser.add_argument("--open-carla-display", help="Display CARLA simulator? Need --debug", default=False,
                            action="store_true")
    arg_parser.add_argument("--track-ego", default=False, action="store_true",
                            help="Track the ego in CARLA simulator? Need --debug and --open-carla-display")

    args = arg_parser.parse_args()
    ego_json = json.load(open(args.ego_json))
    npcs_json = json.load(open(args.npcs_json))
    walkers_json = json.load(open(args.walkers_json))
    weather_json = json.load(open(args.weather_json))
    town = args.town
    save_path = args.save_path
    carla_recorder_path = os.path.abspath(args.carla_recorder_path)
    oracles_dir = args.oracles_dir
    ads = args.ads
    #####################################################################################
    is_debug = args.debug
    open_carla_display = args.open_carla_display
    track_ego = args.track_ego
    #####################################################################################
    is_connect = False
    while not is_connect:
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(10.0)
            client.reload_world(True)
            world = client.load_world(town)
            world_settings = world.get_settings()
            world_settings.synchronous_mode = True
            world_settings.fixed_delta_seconds = 1 / 20
            world.apply_settings(world_settings)
            is_connect = True
            print("Connect to CARLA!")
        except Exception as e:
            print(f"Connect Error in {e}")
            is_connect = False
            subprocess.Popen(
                ["bash", "run_carla.sh", "2000", str(open_carla_display)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("wait for CARLA...")
            time.sleep(10)
            print("Reconnect CARLA...")
    #####################################################################################
    # Leaderboard config
    #####################################################################################
    if ads == "interfuser":
        CarlaDataProvider_interfuser.set_client(client)
        CarlaDataProvider_interfuser.set_world(world)
    #####################################################################################
    # Weather
    #####################################################################################
    weather = world.get_weather()
    weather.cloudiness = weather_json["cloud"]
    weather.precipitation = weather_json["rain"]
    weather.precipitation_deposits = weather_json["puddle"]
    weather.wetness = weather_json["wetness"]
    weather.wind_intensity = weather_json["wind"]
    weather.fog_density = weather_json["fog"]
    weather.sun_azimuth_angle = weather_json["angle"]
    weather.sun_altitude_angle = weather_json["altitude"]
    world.set_weather(weather)
    #####################################################################################
    # Blueprint of CARLA
    #####################################################################################
    carla_collision_sensor_bp = world.get_blueprint_library().find(
        "sensor.other.collision"
    )
    carla_lane_invasion_sensor_bp = world.get_blueprint_library().find(
        "sensor.other.lane_invasion"
    )
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    walker_bp = world.get_blueprint_library().find("walker.pedestrian.0001")  # 0001~0014
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    rgb_camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    rgb_camera_bp.set_attribute("image_size_x", "800")
    rgb_camera_bp.set_attribute("image_size_y", "600")
    rgb_camera_bp.set_attribute("fov", "105")
    #####################################################################################
    # List for record vehicle and its controller
    #####################################################################################
    vehicle_list = []
    #####################################################################################
    # State
    #####################################################################################
    state = SimpleNamespace()

    state.first_frame_id = 0
    state.num_frames = 0

    state.early_stop = False  # early stop if collision etc.
    state.early_stop_reason = ""

    state.violation_found = False

    state.crashed = False
    state.collision_details = []

    state.laneinvaded = False
    state.laneinvasion_details = []

    state.on_red = False
    state.on_red_speed = list()
    state.red_violation = False
    state.running_red_light_details = []

    state.speed_lim = []
    state.speeding = False
    state.speeding_details = []

    state.stuck_duration = 0
    state.stuck = False
    state.stuck_details = []

    #####################################################################################
    # Spawn ego vehicle
    #####################################################################################
    ego_sp, ego_dp = json_2_transform(ego_json, only_sp=False)
    ego_agent = None
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if ego is None:
        print("Error in spawn ego")
        sys.exit(-1)
    world.tick()  # spawn ego vehicle
    if weather_json["altitude"] < 0:  # open light if night
        ego.set_light_state(
            carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
    ego.set_simulate_physics()
    if ads == "interfuser":
        ego_agent = InterfuserAgent("InterFuser/team_code/interfuser_config.py")
        trajectory = [transform_2_location(ego_sp), transform_2_location(ego_dp)]
        gps_route, route = interpolate_trajectory_interfuser(world, trajectory)
        ego_agent.set_global_plan(gps_route, route)
        agent_wrapper = AgentWrapper_interfuser(ego_agent)
        agent_wrapper.setup_sensors(ego)
    elif ads == "behavior":
        ego_agent = BehaviorAgent(vehicle=ego, behavior="cautious")
        ego_agent.set_destination(start_location=ego_sp.location, end_location=ego_dp.location)
        ego_agent._update_information()
    vehicle_list.append(("ego", ego, ego_agent))
    print("Spawn Ego vehicle", ego)
    #####################################################################################
    # Spawn collision and lane invasion sensor
    ####################################################################################
    carla_collision_sensor = world.spawn_actor(
        carla_collision_sensor_bp, carla.Transform(), attach_to=ego
    )
    carla_collision_sensor.listen(
        lambda event: _on_collision(event, state=state)
    )
    world.tick()
    carla_lane_invasion_sensor = world.spawn_actor(
        carla_lane_invasion_sensor_bp, carla.Transform(), attach_to=ego
    )

    carla_lane_invasion_sensor.listen(
        lambda event: _on_invasion(event, state=state)
    )
    world.tick()
    #####################################################################################
    # Spawn RGB camera for creating video
    ####################################################################################
    if os.path.exists("/tmp/fuzzerdata"):
        shutil.rmtree("/tmp/fuzzerdata")
    os.mkdir("/tmp/fuzzerdata")
    camera_tf = carla.Transform(carla.Location(z=1.8))
    camera_front = world.spawn_actor(
        rgb_camera_bp,
        camera_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid
    )
    camera_front.listen(lambda image: _on_front_camera_capture(image))
    world.tick()
    camera_tf = carla.Transform(
        carla.Location(z=50.0),
        carla.Rotation(pitch=-90.0)
    )
    camera_top = world.spawn_actor(
        rgb_camera_bp,
        camera_tf,
        attach_to=ego,
        attachment_type=carla.AttachmentType.Rigid
    )

    camera_top.listen(lambda image: _on_top_camera_capture(image))
    world.tick()
    #####################################################################################
    # Spawn NPC vehicle
    #####################################################################################
    for npc_info in npcs_json:
        npc_bp = random.choice(world.get_blueprint_library().filter('vehicle.*.*'))
        npc_sp, npc_dp = json_2_transform(npc_info, only_sp=False)
        npc = world.try_spawn_actor(npc_bp, npc_sp)
        if npc is None:
            print(f"Error in spawn NPC {npc_info}")
            sys.exit(-1)
        world.tick()  # spawn npc vehicle
        npc.set_simulate_physics()
        npc_agent = BehaviorAgent(vehicle=npc, behavior="aggressive")
        npc_agent.set_destination(start_location=npc_sp.location, end_location=npc_dp.location)
        npc_agent._update_information()
        vehicle_list.append(("npc", npc, npc_agent))
        print("Spawn NPC vehicle", npc)
    #####################################################################################
    # Spawn walkers
    #####################################################################################
    for walker_info in walkers_json:
        walker_sp = json_2_transform(walker_info, only_sp=True)
        walker = world.try_spawn_actor(walker_bp, walker_sp)
        if walker is None:
            continue
        world.tick()  # generate walker
        walker.set_simulate_physics()
        print("Spawn walker", walker)
        if walker_info["type"] == "LINEAR":
            forward_vec = walker_sp.rotation.get_forward_vector()
            controller_walker = carla.WalkerControl()
            controller_walker.direction = forward_vec
            controller_walker.speed = walker_info["speed"]
            walker.apply_control(controller_walker)
        elif walker_info["type"] == "AUTOPILOT":
            print("Want to spawn autopilot walker")
            try:
                _, walker_dp = json_2_transform(walker_info, only_sp=False)
                controller_walker = world.spawn_actor(
                    walker_controller_bp,
                    walker_sp,
                    walker)
                world.tick()  # spawn the controller
                controller_walker.start()
                controller_walker.go_to_location(walker_dp.location)
                controller_walker.set_max_speed(float(walker_info["speed"]))
            except Exception as e:
                print(f"Error in spawn autopilot walker:\n {e}")
                continue
    #####################################################################################
    # CARLA recorder
    #####################################################################################
    recorder = client.start_recorder(
        "/tmp/recorder.rec",
        True,
    )  # the path is in CARLA docker
    #####################################################################################
    # Infos simadfuzz need
    #####################################################################################
    start_time = time.time()
    dur_time = time.time() - start_time
    collect_infos = []
    min_dis = 999999
    #####################################################################################
    print("Start simulation")
    snapshot0 = world.get_snapshot()
    state.first_frame_id = snapshot0.frame
    state.num_frames = 0
    ##################################################
    while dur_time < 60 * 10:  # the max simulation time
        world.tick()
        if state.early_stop:
            print(f"Early Stop! Because of {state.early_stop_reason}")
            break
        dur_time = time.time() - start_time
        snapshot = world.get_snapshot()
        timestamp = snapshot.timestamp
        cur_frame_id = snapshot.frame
        state.num_frames = cur_frame_id - state.first_frame_id
        if ads == "interfuser":
            GameTime_interfuser.on_carla_tick(timestamp)
        #######################################################
        # Run each vehicle controller and collect feedback infos
        #######################################################
        cur_infos = {
            "dur_time": dur_time
        }
        for vehicle_idx, (player_role, player, player_controller) in enumerate(vehicle_list):
            if player_role == "ego":
                if ads == "interfuser":
                    control = player_controller()
                elif ads == "behavior":
                    player_controller._update_information()
                    control = player_controller.run_step()
            else:  # for NPC, all is behavior
                try:
                    player_controller._update_information()
                    control = player_controller.run_step()
                except Exception as npc_behavior_e:  # npc has no more route
                    control = carla.VehicleControl()
            player.apply_control(control)
            player_transform = player.get_transform()
            player_location = player_transform.location
            player_speed = player.get_velocity()
            player_acc = player.get_acceleration()
            player_heading = player_transform.rotation.yaw
            cur_infos.update({
                f"loc_x_{vehicle_idx}": player_location.x,
                f"loc_y_{vehicle_idx}": player_location.y,
                f"speed_x_{vehicle_idx}": player_speed.x,
                f"speed_y_{vehicle_idx}": player_speed.y,
                f"acc_x_{vehicle_idx}": player_acc.x,
                f"acc_y_{vehicle_idx}": player_acc.y,
                f"heading_{vehicle_idx}": player_heading
            })
        collect_infos.append(cur_infos)
        #############################################################################
        # Ego base infos
        #############################################################################
        ego_location = ego.get_location()
        ego_transform = ego.get_transform()
        ego_speed = ego.get_velocity()
        ego_speed2 = 3.6 * math.sqrt(ego_speed.x ** 2 + ego_speed.y ** 2 + ego_speed.z ** 2)
        speed_limit = ego.get_speed_limit()
        #############################################################################
        # Check goal reach the destination
        #############################################################################
        dist_to_goal = ego_location.distance(ego_dp.location)
        if dist_to_goal < 2:
            state.early_stop = True
            state.early_stop_reason = "Ego reach the destination"
        #############################################################################
        # Min distance
        #############################################################################
        for vehicle_idx, (player_role, player, player_controller) in enumerate(vehicle_list):
            if player_role == "ego":
                continue
            dis = ego.get_location().distance(player.get_location())
            if dis < min_dis:
                min_dis = dis
        #############################################################################
        # Running Red Light
        #############################################################################
        if ego.is_at_traffic_light():
            traffic_light = ego.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                if state.on_red:
                    state.on_red_speed.append(ego_speed2)
                else:
                    state.on_red = True
                    state.on_red_speed = list()
        else:
            if state.on_red:
                state.on_red = False
                stopped_at_red = False
                for i, ors in enumerate(state.on_red_speed):
                    if ors < 0.1:
                        stopped_at_red = True
                if not stopped_at_red:
                    state.red_violation = True
                    state.running_red_light_details.append((timestamp, ego_transform))
                    state.violation_found = True
        #############################################################################
        # Speeding
        #############################################################################
        try:
            last_speed_limit = state.speed_lim[-1]
        except:
            last_speed_limit = 0
        if speed_limit != last_speed_limit:
            frame_speed_lim_changed = cur_frame_id
        T = 3  # 0 for strict checking
        if (ego_speed2 > speed_limit and
                cur_frame_id > frame_speed_lim_changed + T * 20):
            state.speeding = True
            state.speeding_details.append((timestamp, ego_transform))
            state.violation_found = True
        #############################################################################
        # Stuck
        #############################################################################
        if ego_speed2 < 1:  # km/h
            state.stuck_duration += 1
        else:
            state.stuck_duration = 0
        if state.stuck_duration > (60 * 20):
            state.stuck = True
            state.stuck_details.append((timestamp, ego_transform))
            state.early_stop = True
            state.early_stop_reason = "Ego stuck too long"
            state.violation_found = True
        #############################################################################
        # Check goal
        #############################################################################
        if ads == "interfuser":
            if len(ego_agent._route_planner.route) == 0:
                state.early_stop = True
                state.early_stop_reason = "Ego has no more route"
        if ads == "behavior":
            if len(ego_agent.get_local_planner()._waypoints_queue) == 0:
                state.early_stop = True
                state.early_stop_reason = "Ego has no more waypoints"
        #############################################################################
        # Track ego in CARLA
        #############################################################################
        if open_carla_display and is_debug and track_ego:
            spectator = world.get_spectator()
            camera_location = ego_location + carla.Location(z=20)  # 设置摄像机在车辆顶部
            camera_rotation = carla.Rotation(pitch=-90)  # 使摄像机朝下
            spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    ##################################################
    client.stop_recorder()
    os.system(f"docker cp carla-sim-2000:/tmp/recorder.rec {carla_recorder_path}")
    os.system(f"docker exec carla-sim-2000 rm /tmp/recorder.rec")
    ##################################################
    if state.violation_found:
        save_video(video_dir=oracles_dir)
    ##################################################
    df = pd.DataFrame(collect_infos)
    df.to_csv(save_path, index=False)
    ##################################################
    json.dump(min_dis, open(f"{oracles_dir}/min_dis.json", "w"))
    ##################################################
    dill.dump(state, open(f"{oracles_dir}/state.pick", "wb"))
    ##################################################
    print(state)
    print("Finished sim_executor.py")
