import json
from collections import deque
import math

import carla
import time
import os
import subprocess

from InterFuser.team_code.interfuser_agent import InterfuserAgent
from agents.navigation.behavior_agent import BehaviorAgent
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


def transform_2_location(transform):
    return carla.Location(
        x=transform.location.x, y=transform.location.y, z=transform.location.z
    )


def multi_point(waypoints_list):
    wps = waypoints_list
    if isinstance(wps, deque) or isinstance(wps[0], tuple):
        wps = [t[0] for t in wps]
    for w_id, w in enumerate(wps):
        if isinstance(w, carla.Transform):
            world.debug.draw_string(
                w.location,  # 位置
                f"WP{w_id}",  # 显示的文本
                draw_shadow=True,  # 是否显示阴影
                color=carla.Color(r=255, g=0, b=0),  # 颜色（红色）
                life_time=10000,  # 显示时间（秒），设为永久可设为0，但需每帧刷新
            )
        if isinstance(w, carla.Waypoint):
            world.debug.draw_string(
                w.transform.location,  # 位置
                f"WP{w_id}",  # 显示的文本
                draw_shadow=True,  # 是否显示阴影
                color=carla.Color(r=255, g=0, b=0),  # 颜色（红色）
                life_time=10000,  # 显示时间（秒），设为永久可设为0，但需每帧刷新
            )
    world.tick()


def point(waypoint, message="P"):
    if isinstance(waypoint, carla.Waypoint):
        loc = waypoint.transform.location
    if isinstance(waypoint, carla.Transform):
        loc = waypoint.location
    if isinstance(waypoint, tuple):
        loc = waypoint[0].transform.location
    world.debug.draw_string(
        loc,  # 位置
        message,  # 显示的文本
        draw_shadow=True,  # 是否显示阴影
        color=carla.Color(r=255, g=0, b=0),  # 颜色（红色）
        life_time=10000,  # 显示时间（秒），设为永久可设为0，但需每帧刷新
    )
    world.tick()


subprocess.Popen(["./run_carla.sh", "2000", "True"])
time.sleep(10)
client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world("Town03")
setting = world.get_settings()
setting.synchronous_mode = True
setting.fixed_delta_seconds = 1 / 20
world.apply_settings(setting)
world.tick()
spec = world.get_spectator()
spec.set_transform(
    carla.Transform(
        carla.Location(x=13.968930, y=-183.642151, z=60.785732),
        carla.Rotation(pitch=-88.998283, yaw=-87.767410, roll=0.001563),
    )
)

ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
ego_bp.set_attribute("color", "255,0,0")

right_car_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
right_car_bp.set_attribute("color", "0,255,0")

left_car_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
left_car_bp.set_attribute("color", "255,255,255")

for repeat_index in range(1):
    s1_path = None
    s2_path = None
    for s in ["s1", "s2"]:
        subprocess.Popen(["./run_carla.sh", "2000", "True"])
        time.sleep(10)
        client = carla.Client("localhost", 2000)
        client.set_timeout(10)
        world = client.load_world("Town03")
        setting = world.get_settings()
        setting.synchronous_mode = True
        setting.fixed_delta_seconds = 1 / 20
        world.apply_settings(setting)
        world.tick()
        spec = world.get_spectator()
        spec.set_transform(
            carla.Transform(
                carla.Location(x=13.968930, y=-183.642151, z=60.785732),
                carla.Rotation(pitch=-88.998283, yaw=-87.767410, roll=0.001563),
            )
        )
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(1920))
        camera_bp.set_attribute('image_size_y', str(1080))
        camera_bp.set_attribute('fov', '90')
        camera_location = carla.Location(x=13.968930, y=-183.642151, z=60.785732)
        camera_rotation = carla.Rotation(pitch=-88.998283, yaw=-87.767410, roll=0.001563)
        camera_transform = carla.Transform(camera_location, camera_rotation)
        traffic_lights = world.get_actors().filter("traffic.traffic_light")

        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(1000)
        if s == "s1":
            ego_sp = carla.Transform(
                carla.Location(x=-28.760317, y=-194.588547, z=2.5),
                carla.Rotation(pitch=9.061202, yaw=7.022564, roll=0.000001),
            )
            ego_dp = carla.Transform(
                carla.Location(x=44.028740, y=-192.691544, z=0.5),
                carla.Rotation(pitch=-3.409507, yaw=7.079691, roll=0.000000),
            )

            right_car_sp = carla.Transform(
                carla.Location(x=10.490709, y=-140.834000, z=2.5),
                carla.Rotation(pitch=0.000000, yaw=-88.586456, roll=0.000000),
            )
            right_car_dp = carla.Transform(
                carla.Location(x=55.028740, y=-192.691544, z=0.5),
                carla.Rotation(pitch=-3.409507, yaw=7.079691, roll=0.000000),
            )

            left_car_sp = carla.Transform(
                carla.Location(x=35.411037, y=-203.848938, z=2.5),
                carla.Rotation(pitch=9.214786, yaw=-178.001694, roll=0.000005),
            )
            left_car_dp = carla.Transform(
                carla.Location(x=-0.156147, y=-175.754395, z=0.5),
                carla.Rotation(pitch=-4.225831, yaw=92.103363, roll=0.000001),
            )
        if s == "s2":
            ego_sp = carla.Transform(
                carla.Location(x=-28.760317, y=-194.588547, z=2.5),
                carla.Rotation(pitch=9.061202, yaw=7.022564, roll=0.000001),
            )
            ego_dp = carla.Transform(
                carla.Location(x=44.028740, y=-192.691544, z=0.5),
                carla.Rotation(pitch=-3.409507, yaw=7.079691, roll=0.000000),
            )

            right_car_sp = carla.Transform(
                carla.Location(x=10.490709, y=-130.034000, z=2.5),
                carla.Rotation(pitch=0.000000, yaw=-88.586456, roll=0.000000),
            )
            right_car_dp = carla.Transform(
                carla.Location(x=55.028740, y=-192.691544, z=0.5),
                carla.Rotation(pitch=-3.409507, yaw=7.079691, roll=0.000000),
            )

            left_car_sp = carla.Transform(
                carla.Location(x=35.411037, y=-203.848938, z=2.5),
                carla.Rotation(pitch=9.214786, yaw=-178.001694, roll=0.000005),
            )
            left_car_dp = carla.Transform(
                carla.Location(x=-0.156147, y=-175.754395, z=0.5),
                carla.Rotation(pitch=-4.225831, yaw=92.103363, roll=0.000001),
            )

        ego = world.spawn_actor(ego_bp, ego_sp)
        right_car = world.spawn_actor(right_car_bp, right_car_sp)
        left_car = world.spawn_actor(left_car_bp, left_car_sp)
        physics_control = ego.get_physics_control()
        max_steer_angle = 0
        for wheel in physics_control.wheels:
            if wheel.max_steer_angle > max_steer_angle:
                max_steer_angle = wheel.max_steer_angle

        CarlaDataProvider_interfuser.set_client(client)
        CarlaDataProvider_interfuser.set_world(world)
        GameTime_interfuser.restart()
        ego_agent = InterfuserAgent("InterFuser/team_code/interfuser_config.py")
        trajectory = [transform_2_location(ego_sp), transform_2_location(ego_dp)]
        gps_route, route = interpolate_trajectory_interfuser(world, trajectory)
        ego_agent.set_global_plan(gps_route, route)
        agent_wrapper = AgentWrapper_interfuser(ego_agent)
        agent_wrapper.setup_sensors(ego)
        world.tick()

        right_car_agent = BehaviorAgent(right_car, behavior="normal")
        right_route_right = right_car_agent.trace_route(
            world.get_map().get_waypoint(right_car_sp.location),
            world.get_map().get_waypoint(right_car_dp.location),
        )
        right_car_agent.set_global_plan(right_route_right)

        left_car_agent = BehaviorAgent(left_car, behavior="normal")
        left_route_right = right_car_agent.trace_route(
            world.get_map().get_waypoint(left_car_sp.location),
            world.get_map().get_waypoint(left_car_dp.location),
        )
        left_car_agent.set_global_plan(left_route_right)
        ####
        camera = world.spawn_actor(camera_bp, camera_transform)
        def save_image(image, s_name):
            filename = f"s_images/{s_name}_{image.frame}.jpg"
            image.save_to_disk(filename)
        camera.listen(lambda image: save_image(image, s))
        ####
        while True:
            world.tick()
            snapshot = world.get_snapshot()
            snapshot_timestamp = snapshot.timestamp
            GameTime_interfuser.on_carla_tick(snapshot_timestamp)
            if not right_car_agent.done():
                right_control = right_car_agent.run_step()
                right_car.apply_control(right_control)
            if not left_car_agent.done():
                left_control = left_car_agent.run_step()
                left_car.apply_control(left_control)
            ego_control = ego_agent()
            ego.apply_control(ego_control)
            if ego.get_location().distance(ego_dp.location) < 5:
                break
        
        print("Finished")
        left_car.destroy()
        right_car.destroy()
        ego_agent.destroy()
        agent_wrapper.cleanup()
        ego.destroy()
        CarlaDataProvider_interfuser.cleanup()
        s_images=f"ls ./s_images/{s}_*.jpg "+"| sort -V | xargs -I {} cat {}"
        vid_filename = f"./s_images/{s}.mp4"
        cmd_ffmpeg = " ".join([
            "ffmpeg",
            "-f image2pipe",
            f"-r 20",
            "-vcodec mjpeg",
            "-i -",
            "-vcodec libx264",
            vid_filename
        ])

        cmd = f"{s_images} | {cmd_ffmpeg} 2> /dev/null"
        os.system(cmd)
        os.system("rm ./s_images/*.jpg")
