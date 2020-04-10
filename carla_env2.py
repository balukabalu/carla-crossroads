import glob
import os
import sys
import numpy as np
import math
from scipy.spatial import distance

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print("eeee")
except IndexError:
    print("fffffffffff")
    pass



import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref


import time




class CarlaEnv(object):

    def __init__(self):
        self.actor_list = []
        self.client = carla.Client('localhost', 2000)
        self.world = None
        self.blueprint_library = None
        self.vehicle_bp = None
        self.vehicles = []
        self.camera_bp = None
        self.camera = None
        self.current_map = None
        self.lane_ids = None
        self.col_sensor = []

        self.lane_width = None
        self.spectator = None
        self.col_sensor = []
        self.collosion_event = False

        self.spawn_points = []
        self.nulla = False 


    def kill(self):

        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        print('done.')


    def spect_cam(self, vehicle):
        self.spectator = self.world.get_spectator()
        tr = vehicle.get_transform()
        tr.location.z+=80
        tr.location.x=-151
        tr.location.y=-35
        wp = self.current_map.get_waypoint(vehicle.get_transform().location,project_to_road=True, lane_type=carla.LaneType.Driving)
        tr.rotation = carla.Rotation(pitch=-90.000000, yaw=-180.289116, roll=0.000000)
        self.spectator.set_transform(tr)


    def collevent(self, event):

        eventframe = event.frame
        eventactor = event.actor
        eventotheractor = event.other_actor
        for c in range(0,4):
            if eventotheractor.id == self.vehicles[c].id:
                    print("COLLISION!!!!!!!!!!!")
                    self.collosion_event = True
        print (eventframe, "      ", eventactor, "       ", eventotheractor)


    def start2(self):
        if (True):
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(40.0)  
            self.client.load_world( '/Game/Carla/Maps/Town07')     
            self.world = self.client.get_world()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            #weather = carla.WeatherParameters(cloudyness = 20)
            #self.world.set_weather(weather)
            self.current_map = self.world.get_map()
            self.blueprint_library = self.world.get_blueprint_library()

            self.spawn_points.append( carla.Transform(carla.Location(x=-152.758665, y=-75.552750, z=1.270000), carla.Rotation(pitch=0.000000, yaw=90.289116, roll=0.000000)))
            self.spawn_points.append( carla.Transform(carla.Location(x=-148.758665, y=5.552750, z=1.270000), carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000)))
            self.spawn_points.append(carla.Transform(carla.Location(x=-110.758665, y=-37.552750, z=1.270000), carla.Rotation(pitch=0.000000, yaw=-180.289116, roll=0.000000)))
            self.spawn_points.append( carla.Transform(carla.Location(x=-190.758665, y=-32.552750, z=1.270000), carla.Rotation(pitch=0.000000, yaw=0.289116, roll=0.000000))
)


    def setup_car(self):		#AUTÓ LÉTREHOZÁSA
       
        print("Spawns actor-vehicle to be controled.")
        self.actor_list = []
        self.col_sensor = []
        self.collosion_event = False
        
        self.vehicle_bp = self.blueprint_library.find('vehicle.audi.etron')
        self.vehicle_bp.set_attribute('color', '255,20,147')
        self.vehicles = []
        self.vehicles.append(self.world.spawn_actor(self.vehicle_bp, self.spawn_points[0]))
        self.vehicles.append(self.world.spawn_actor(self.vehicle_bp, self.spawn_points[1]))
        self.vehicles.append(self.world.spawn_actor(self.vehicle_bp, self.spawn_points[2]))
        self.vehicles.append(self.world.spawn_actor(self.vehicle_bp, self.spawn_points[3]))

#------------------------------------------------------
# collision sensor 
#------------------------------------------
        print("COL")
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        for c in range(0,4):
            print("C")
            self.col_sensor.append(self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicles[c]))
            print("CC")
            self.col_sensor[c].listen(lambda event: self.collevent(event))
#---------------------------------------------------------------
        print("B")
        self.actor_list.append(self.vehicles[0])
        self.actor_list.append(self.vehicles[1])
        self.actor_list.append(self.vehicles[2])
        self.actor_list.append(self.vehicles[3])

        print("A")






    def snap_to_s_t(self, world_snapshot, currentmap, vehicle):	#OB FÜGGVÉNY

        vehicle_snapshot = world_snapshot.find(vehicle.id)

        vehicle_transform = vehicle_snapshot.get_transform()
        vehicle_velocity = vehicle_snapshot.get_velocity()
        vehicle_ang_vel = vehicle_snapshot.get_angular_velocity()
        vehicle_acceleration = vehicle_snapshot.get_acceleration()

        road_waypoint = None
        road_waypoint = currentmap.get_waypoint(vehicle_transform.location,project_to_road=True, lane_type=carla.LaneType.Driving)


        p_veh = (vehicle_transform.location.x, vehicle_transform.location.y)
        p_road = (road_waypoint.transform.location.x, road_waypoint.transform.location.y)
        v_veh_road = (p_veh[0] - p_road[0], p_veh[1] - p_road[1])
        v_road = (math.cos(road_waypoint.transform.rotation.yaw*math.pi/180), math.sin(road_waypoint.transform.rotation.yaw*math.pi/180))

        lat_err = -distance.euclidean(p_veh, p_road)*np.sign(np.cross(v_veh_road, v_road))
        ang_err = vehicle_transform.rotation.yaw - road_waypoint.transform.rotation.yaw
        if ang_err > 180:
            ang_err = -360+ang_err
        if ang_err < -180:
            ang_err = 360+ang_err
        speed = math.sqrt(vehicle_velocity.x**2 +  vehicle_velocity.y**2)
        acc = math.sqrt(vehicle_acceleration.x**2 + vehicle_acceleration.y**2)
        speedX = abs(speed*math.cos(vehicle_transform.rotation.yaw*math.pi/180-math.atan2(vehicle_velocity.y, vehicle_velocity.x)))
        speedY = speed*math.sin(vehicle_transform.rotation.yaw*math.pi/180-math.atan2(vehicle_velocity.y, vehicle_velocity.x))
        ang_vel = vehicle_ang_vel.z
        accX = acc*math.cos(vehicle_transform.rotation.yaw*math.pi/180-math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))
        accY = acc*math.sin(vehicle_transform.rotation.yaw*math.pi/180-math.atan2(vehicle_acceleration.y, vehicle_acceleration.x))

        if speedX < 2:
            self.nulla = True
        else:
            self.nulla = False 

        return np.hstack((lat_err, ang_err, speedX, speedY, ang_vel, accX, accY))

    def stanley(self, world_snapshot, c):

        ob= self.snap_to_s_t(world_snapshot, self.current_map, self.vehicles[c])
        vehicle_snapshot = world_snapshot.find(self.vehicles[c].id)
        vehicle_transform = vehicle_snapshot.get_transform()
        k = 3

        if c == 1 and vehicle_transform.location.y < -27 and vehicle_transform.location.y > -37 and vehicle_transform.location.x > -159: 

            k, ob[0], ob[1] = self.curve_stanley(-159, -27, vehicle_transform)
        if c ==0 and vehicle_transform.location.y < -33 and vehicle_transform.location.y > -43 and vehicle_transform.location.x < -143: 
            k, ob[0], ob[1] = self.curve_stanley(-141.5, -43, vehicle_transform)
            print(vehicle_transform.location.y)

        steering_angle = -ob[1]*math.pi/180 - math.atan2(k*(  ob[0]), ob[2])
        return steering_angle, ob

    def curve_stanley(self, c1, c2, t):
        
        dist = distance.euclidean((c1, c2),(t.location.x, t.location.y))-10
        v1 = (c1-t.location.x, -c2 + t.location.y)
        v2 = (math.cos(t.rotation.yaw*math.pi/180), math.sin(t.rotation.yaw*math.pi/180))
        angle = math.acos(np.dot(v1, v2) / (math.sqrt(np.dot(v1, v1)) * math.sqrt(np.dot(v2,v2))))
        return 10, dist,angle



    def snapshot_to_position(self, world_snapshot):
        vehicle1_snapshot = world_snapshot.find(self.vehicles[0].id)
        vehicle1_transform = vehicle1_snapshot.get_transform()
        vehicle2_snapshot = world_snapshot.find(self.vehicles[1].id)
        vehicle2_transform = vehicle2_snapshot.get_transform()
        vehicle3_snapshot = world_snapshot.find(self.vehicles[2].id)
        vehicle3_transform = vehicle3_snapshot.get_transform()
        vehicle4_snapshot = world_snapshot.find(self.vehicles[3].id)
        vehicle4_transform = vehicle4_snapshot.get_transform()

        v1x = vehicle1_transform.location.x
        v1y = vehicle1_transform.location.y
        v2x = vehicle2_transform.location.x
        v2y = vehicle2_transform.location.y
        v3x = vehicle3_transform.location.x
        v3y = vehicle3_transform.location.y
        v4x = vehicle4_transform.location.x
        v4y = vehicle4_transform.location.y

        return np.hstack((v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y))

    def step(self, u, j):

        ob = np.hstack((None, None, None, None))
        a = self.world.tick()

        self.spect_cam(self.vehicles[1])
        world_snapshot = self.world.get_snapshot()
        control = np.hstack((carla.VehicleControl(), carla.VehicleControl(), carla.VehicleControl(), carla.VehicleControl()))



        for c in range(len(control)):
            
            control[c].steer, ob[c] = self.stanley(world_snapshot, c)
            control[c].throttle =  max(0, float(u[c]))
            if j<100:
                control[c].throttle = 0.5
                control[c].brake = 0
                control[c].steer = 0


        for c in range(len(control)):
            self.vehicles[c].apply_control(control[c])


        reward = 0
        done = False
        obbb = self.snapshot_to_position(world_snapshot)
        if self.collosion_event == True:
            done = True
            reward = -100.0
        if obbb[0] > -143 and obbb[2] < -159 and obbb[4] < -159 and obbb[6] > -143:
            done = True
            reward = 100.0

        if j > 80 and self.nulla == True:
            done = True
            reward = -100.0


        print(reward)

        if False: #	done:
            for c in range(len(self.vehicles)):
                sp = self.spawn_points[c]
                if c < 2: 
                    sp.location.y += -10+ random.random()*20
                else:
                    sp.location.x += -10 + random.random()*20
                self.vehicles[c].set_transform(sp)            
     


        return obbb, reward, done
        


