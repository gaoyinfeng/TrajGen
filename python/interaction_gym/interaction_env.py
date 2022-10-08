#-*- coding: UTF-8 -*- 
import sys
sys.path.append("..")

import os
import glob
import copy
import numpy as np
import math
import argparse
import zmq

import lanelet2
import lanelet2_matching

import geometry
import reward

from interaction_map import InteractionMap
from ego_vehicle import EgoVehicle
from observation import Observation
from interaction_rl.config import hyperParameters

class InteractionEnv:
    def __init__(self, setting):
        self._continous_action = setting['continous_action']
        self._control_steering = setting['control_steering']
        self._route_type = setting['route_type']

        self._visualaztion = setting['visualaztion']
        self._ghost_visualaztion = setting['ghost_visualaztion']
        self._route_visualaztion = setting['route_visualaztion']
        self._route_bound_visualaztion = setting['route_bound_visualaztion']

        self._config = hyperParameters(self._control_steering)

        self._map = InteractionMap(setting)  # load map and track file

        self._delta_time = 100 # 100 ms
        self._start_end_state = None     # ego vehicle start & end state (start_time,end_time,length,width,start motion_state,end motion_state)
        self._ego_vehicle_dict = dict()
        self._ego_route_dict = dict()
        
        self._ego_previous_route_points_dict = dict()
        self._ego_future_route_points_dict = dict()
        self._ego_trajectory_record = dict()

        self._scenario_start_time = None
        self._scenario_end_time = None # select the earliest end time
        self._stepnum = 0
       
        self._observation = None


    def __del__(self):
        self._map.__del__()


    def change_predict_track_file(self, trajectory_file_name=None):
        self._map.change_predict_track_file(trajectory_file_name = trajectory_file_name)

        
    def change_ground_truth_track_file(self, track_file_number=None):
        self._map.change_ground_truth_track_file(track_file_number = track_file_number)


    def choose_ego_and_init_map(self, ego_info_dict):
        print('')
        print('init map and choose ego:')
        # map init
        self._map.map_init()
        if ego_info_dict['ego_id_list'] is None or len(ego_info_dict['ego_id_list']) == 0:
            print('random choose ego')
            self._map.random_choose_ego_vehicle()
        else:
            print('specify choose ego')
            self._map.specify_id_choose_ego_vehicle(ego_info_dict['ego_id_list'], ego_info_dict['ego_start_timestamp'])

        # ego init
        self._ego_vehicle_dict.clear()
        self._ego_route_dict.clear()
        self._start_end_state = self._map._ego_vehicle_start_end_state_dict  # ego vehicle start & end state dict, key = ego_id, value = (start_time,end_time,length,width,start motion_state,end motion_state)
        if self._route_type == 'predict':
            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = EgoVehicle(start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = ego_info_dict['ego_route'][ego_id]

            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._scenario_end_time = self._scenario_start_time + 100 * self._config.max_steps # 10s = 100 * 0.1s

        elif self._route_type == 'ground_truth':
            self._ground_truth_route = self.get_ground_truth_route(ego_info_dict['ego_start_timestamp'])
            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = EgoVehicle(start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = self._ground_truth_route[ego_id]
            
            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._scenario_end_time = min([self._start_end_state[i][1] for i in self._start_end_state]) # select the earliest end time among all ego vehicle

        elif self._route_type == 'centerline':
            self._centerline_route = self.get_centerline_route(ego_info_dict['ego_start_timestamp'])
            for ego_id, start_end_state in self._start_end_state.items():
                self._ego_vehicle_dict[ego_id] = EgoVehicle(start_end_state, self._delta_time) # delta_time means tick-time length
                self._ego_route_dict[ego_id] = self._centerline_route[ego_id]
            
            self._scenario_start_time = max([self._start_end_state[i][0] for i in self._start_end_state]) # select the latest start time among all ego vehicle
            self._scenario_end_time = min([self._start_end_state[i][1] for i in self._start_end_state]) # select the earliest end time among all ego vehicle

        if self._scenario_start_time > self._scenario_end_time:
            print('start time > end time?')
            return False

        # ego observation (manager) init
        self._observation = Observation(self._ego_vehicle_dict, self._map, self._config, self._control_steering)
        return self._map._ego_vehicle_id_list


    def reset(self):
        print('')
        print('reset')
        # reset ego state and trajectory record
        ego_state_dict = dict()
        self._ego_trajectory_record.clear()
        self._ego_previous_route_points_dict.clear()
        self._ego_previous_steer = 0 # for calculate steer reward

        for ego_id, ego_state in self._start_end_state.items():
            print('ego vehicle id:', ego_id)
            ego_state_dict[ego_id] = ego_state[4] # now "state" only contains start motion_state: (time_stamp_ms, x, y, vx, vy, psi_rad)
            self._ego_trajectory_record[ego_id] = []
        for ego_id, ego_state in self._ego_vehicle_dict.items():
            print('reset ego state')
            ego_state.reset_state(self._start_end_state[ego_id][4])
            self._ego_trajectory_record[ego_id].append([ego_state._current_state.x, ego_state._current_state.y, ego_state._current_state.vx, ego_state._current_state.vy, ego_state._current_state.psi_rad])
        
        # reset global environment time
        self._current_time = self._scenario_start_time

        # reset/clear observation
        reset_success = self._observation.reset(self._route_type, self._ego_route_dict)
        if not reset_success:
            print('reset failure')
            return None

        # reset episode
        self._stepnum = 0
        self._total_stepnum = (self._scenario_end_time - self._scenario_start_time) / self._delta_time
        self._map.update_param(self._current_time, ego_state_dict)
        
        # get observation
        init_observation_dict = self._observation.get_scalar_observation(self._current_time)
        
        # visualize map, vehicles and ego's route
        self._ego_future_route_points_dict = self._observation.get_future_route_points(init_observation_dict)
        if self._visualaztion:
            # specified ego(with/without ghost ego) and selected vehicle highlight
            interaction_vehicle_id_list = self._observation.get_intersection_vehicle_id(init_observation_dict)
            self._map.render_vehicles(ego_state_dict, interaction_vehicle_id_list, ghost_vis=self._ghost_visualaztion)
            # render ego's route
            if self._route_visualaztion:
                self._map.render_route(self._ego_route_dict)
                self._map.render_future_route_points(self._ego_previous_route_points_dict, self._ego_future_route_points_dict)
                if self._route_bound_visualaztion:
                    self._map.render_route_bounds(self._observation.ego_route_left_bound_points, self._observation.ego_route_right_bound_points)
                    self._previous_bound_points_list = []
                    current_bound_points_list = self._observation.get_current_bound_points(init_observation_dict)
                    self._map.render_closet_bound_point(self._previous_bound_points_list, current_bound_points_list)
                    
        return init_observation_dict


    def step(self, action_dict):
        print('')
        print('step:', self._stepnum, '/', self._total_stepnum)
        
        ego_state_dict = dict()
        ego_action_dict = dict()

        reward_dict = dict()
        aux_info_dict = dict()

        # step ego
        if self._continous_action:
            for ego_id, action_list in action_dict.items():
                if self._control_steering:
                    ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list)
                else: # control target speed
                    future_route_points_list = self._ego_future_route_points_dict[ego_id]
                    index = int(len(future_route_points_list)/2)
                    next_waypoint_position = [future_route_points_list[index][0], future_route_points_list[index][1]]

                    ego_state, ego_action = self._ego_vehicle_dict[ego_id].step_continuous_action(action_list, next_waypoint_position)

                ego_state_dict[ego_id] = ego_state
                ego_action_dict[ego_id] = ego_action
        
        self._current_time += self._delta_time
        self._stepnum += 1
        self._map.update_param(self._current_time, ego_state_dict)

        # get new observation, calculate rewards and results
        next_time_observation_dict = self._observation.get_scalar_observation(self._current_time)
        done_dict, result_dict = self.reach_terminate_condition(self._current_time, next_time_observation_dict)
        for ego_id in result_dict.keys():
            # record result 
            aux_info_dict[ego_id] = dict()
            aux_info_dict[ego_id]['result'] = result_dict[ego_id]
            # record actual trajcetory
            ego_state = ego_state_dict[ego_id]
            self._ego_trajectory_record[ego_id].append([ego_state.x, ego_state.y, ego_state.vx, ego_state.vy, ego_state.psi_rad])
            aux_info_dict[ego_id]['trajectory'] = self._ego_trajectory_record[ego_id]

            # terminal reward
            if aux_info_dict[ego_id]['result'] == 'success':
                terminal_reward = 0
            elif aux_info_dict[ego_id]['result'] == 'time_exceed':
                terminal_reward = 0
            elif aux_info_dict[ego_id]['result'] == 'collision':
                current_speed_norm = next_time_observation_dict['current_speed'][ego_id][0]/10
                terminal_reward = -500 * (1 + current_speed_norm)
            elif aux_info_dict[ego_id]['result'] == 'deflection':
                terminal_reward = 0
            else:
                terminal_reward = 0
            # step reward
            if self._control_steering:
                position_reward = reward.calculate_lane_keeping_reward(next_time_observation_dict, ego_id)
                speed_reward = 0
                steer_reward = reward.calculate_steer_reward(self._ego_previous_steer, ego_action_dict[ego_id].steering)
                self._ego_previous_steer = ego_action_dict[ego_id].steering
            else:
                position_reward = reward.calculate_trajectory_pos_reward(next_time_observation_dict, ego_id)
                speed_reward = 0
                steer_reward = 0 
            
            step_reward = -0.5
            
            env_reward = terminal_reward + position_reward + steer_reward + speed_reward + step_reward
            reward_dict[ego_id] = env_reward

        # visualize map, vehicles and ego's route
        self._ego_future_route_points_dict = self._observation.get_future_route_points(next_time_observation_dict)
        if self._visualaztion:            
            # specified ego(with/without ghost ego) and selected vehicle highlight
            interaction_vehicle_id_list = self._observation.get_intersection_vehicle_id(next_time_observation_dict)
            self._map.render_vehicles(ego_state_dict, interaction_vehicle_id_list, ghost_vis=self._ghost_visualaztion)
            # render ego's route
            if self._route_visualaztion:
                self._map.render_future_route_points(self._ego_previous_route_points_dict, self._ego_future_route_points_dict)
                self._ego_previous_route_points_dict = self._ego_future_route_points_dict
                if self._route_bound_visualaztion:
                    current_bound_points_list = self._observation.get_current_bound_points(next_time_observation_dict)
                    self._map.render_closet_bound_point(self._previous_bound_points_list, current_bound_points_list)
                    self._previous_bound_points_list = current_bound_points_list

        return next_time_observation_dict, reward_dict, done_dict, aux_info_dict

    def reach_terminate_condition(self, current_time, observation_dict):
        done_dict = dict()
        result_dict = dict()

        ego_id_list = observation_dict['ego_shape'].keys()

        # none by default
        for ego_id in ego_id_list:
            done_dict[ego_id] = False
            result_dict[ego_id] = 'none'

        # reach end time
        if not (current_time + self._delta_time < self._scenario_end_time):
            print('END: reach end time')
            for ego_id in ego_id_list:
                done_dict[ego_id] = True
                result_dict[ego_id] = 'time_exceed'
        # success, collision or deflection 
        else:
            for observation_type, observation_content_dict in observation_dict.items():
                # successfully reach end point ego vehicles
                if observation_type == 'reach_end':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'success'
                            print(ego_id, 'Success: reach goal point')
                # collision ego vehicles
                elif observation_type == 'collision':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'collision'
                            print(ego_id, 'Fail: collision')
                # deflection ego vehicles
                elif observation_type == 'deflection':
                    for ego_id in ego_id_list:
                        if observation_content_dict[ego_id] is True:
                            done_dict[ego_id] = True
                            result_dict[ego_id] = 'deflection'
                            print(ego_id, 'Fail: deflection (distance)')

        return done_dict, result_dict

    # generate centerline routes
    def get_centerline_route(self, start_timestamp_list):
        centerline_route_dict = dict()
        track_dict = self._map.track_dict
        for vehicle_id in self._start_end_state.keys():
            vehicle_dict = track_dict[vehicle_id]
            # time horizen
            if len(start_timestamp_list) != 0:
                start_timestamp = int(start_timestamp_list[0])
                end_timestamp = start_timestamp + 100 * self._config.max_steps - 100
            else:
                start_timestamp = vehicle_dict.time_stamp_ms_first
                end_timestamp = vehicle_dict.time_stamp_ms_last
            # in order to get all of the lanelets
            initial_timestamp = vehicle_dict.time_stamp_ms_first
            terminal_timestamp = vehicle_dict.time_stamp_ms_last

            # get vehicle's whole lanelet
            ms_dict = vehicle_dict.motion_states
            start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, initial_timestamp, terminal_timestamp)

            # if cant find proper start and end lanelet, then:
            if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                initial_lanelet, terminal_lanelet = start_lanelet, end_lanelet
                print('can\'t find route, try to use start time instead of initial time')
                start_lanelet, end_lanelet = self.get_start_end_lanelet_from_ms_dict_with_min_heading(ms_dict, start_timestamp, end_timestamp)
                if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    print('still can\'t find route, try to mix them up')
                    start_lanelet, end_lanelet = self.try_to_find_practicable_start_end_lanelet(start_lanelet, initial_lanelet, end_lanelet, terminal_lanelet)
                    if not start_lanelet or not end_lanelet or not self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                        print('the centerline route doesn\'t exist, using ground truth route')
                        ground_truth_route_dict = self.get_ground_truth_route(start_timestamp_list)
                        centerline_route_dict[vehicle_id] = ground_truth_route_dict[vehicle_id]
                        continue

            # if proper start and end lanelet exist
            route_lanelet = self.get_route_lanelet(start_lanelet, end_lanelet)

            # get vehicle's route based on whole route lanelet and a specific time horizen
            vehicle_start_pos = [ms_dict[start_timestamp].x, ms_dict[start_timestamp].y]
            vehicle_end_pos = [ms_dict[end_timestamp].x, ms_dict[end_timestamp].y]
            vehicle_route_list = self.get_route_from_lanelet(route_lanelet, vehicle_start_pos, vehicle_end_pos)
            # if this route's start point away from ego's start position
            start_point = [vehicle_route_list[0][0], vehicle_route_list[0][1]]
            if math.sqrt((start_point[0] - vehicle_start_pos[0])**2 + (start_point[1] - vehicle_start_pos[1])**2) > 3:
                print('the centerline route doesn\'t reliable, using ground truth route')
                ground_truth_route_dict = self.get_ground_truth_route(start_timestamp_list)
                centerline_route_dict[vehicle_id] = ground_truth_route_dict[vehicle_id]
            else:
                centerline_route_dict[vehicle_id] = vehicle_route_list

        return centerline_route_dict
    
    def try_to_find_practicable_start_end_lanelet(self, start_lanelet_1, start_lanelet_2, end_lanelet_1, end_lanelet_2):
        start_list = []
        end_list = []
        if start_lanelet_1:
            start_list.append(start_lanelet_1)
            start_lanelet_3_list = self._map.routing_graph.previous(start_lanelet_1)
            if start_lanelet_3_list:
                start_list.append(start_lanelet_3_list[0])
        if start_lanelet_2:
            start_list.append(start_lanelet_1)
            start_lanelet_4_list = self._map.routing_graph.previous(start_lanelet_2)
            if start_lanelet_4_list:
                start_list.append(start_lanelet_4_list[0])
        if end_lanelet_1:
            end_list.append(end_lanelet_1)
            end_lanelet_3_list = self._map.routing_graph.following(end_lanelet_1)
            if end_lanelet_3_list:
                end_list.append(end_lanelet_3_list[0])
        if end_lanelet_2:
            end_list.append(end_lanelet_2)
            end_lanelet_4_list = self._map.routing_graph.following(end_lanelet_2)
            if end_lanelet_4_list:
                end_list.append(end_lanelet_4_list[0])


        for start_lanelet in start_list:
            for end_lanelet in end_list:
                if self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0):
                    return start_lanelet, end_lanelet
        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # get the start and end lanelet set of ego vehicles
        traffic_rules = self._map.traffic_rules
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)

        obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)

        if len(obj_start_matches_rule_compliant) > 0:
            # first matching principle
            start_lanelet = obj_start_matches_rule_compliant[0].lanelet

        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)

        if obj_end_matches_rule_compliant:
            end_lanelet = obj_end_matches_rule_compliant[0].lanelet

        return start_lanelet, end_lanelet

    def get_start_end_lanelet_from_ms_dict_with_min_heading(self, ms_dict, start_timestamp, end_timestamp):
        start_lanelet = None
        end_lanelet = None

        traffic_rules = self._map.traffic_rules
        
        # start lanelet
        ms_initial = ms_dict[start_timestamp]
        vehicle_initial_pos = (ms_initial.x, ms_initial.y)
        vehicle_initial_velocity = (ms_initial.vx, ms_initial.vy)

        obj_start = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_initial_pos[0], vehicle_initial_pos[1], 0), [])
        obj_start_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_start, 0.2)
        obj_start_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_start_matches, traffic_rules)
        if len(obj_start_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            start_lanelet_index = 0

            for index, match in enumerate(obj_start_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_initial_pos, vehicle_initial_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    start_lanelet_index = index
            start_lanelet = obj_start_matches_rule_compliant[start_lanelet_index].lanelet
        
        # end lanelet
        ms_terminal = ms_dict[end_timestamp]
        vehicle_terminal_pos = (ms_terminal.x, ms_terminal.y)
        vehicle_terminal_velocity = (ms_terminal.vx, ms_terminal.vy)
        
        obj_end = lanelet2_matching.Object2d(1,lanelet2_matching.Pose2d(vehicle_terminal_pos[0], vehicle_terminal_pos[1], 0), [])
        obj_end_matches = lanelet2_matching.getDeterministicMatches(self._map.laneletmap, obj_end, 0.2)
        obj_end_matches_rule_compliant = lanelet2_matching.removeNonRuleCompliantMatches(obj_end_matches, traffic_rules)
        if len(obj_end_matches_rule_compliant) > 0:
            # similar min heading error matching principle
            min_heading_error = 90
            end_lanelet_index = 0

            for index,match in enumerate(obj_end_matches_rule_compliant):
                match_lanelet = match.lanelet
                heading_error = geometry.get_vehicle_and_lanelet_heading_error(vehicle_terminal_pos, vehicle_terminal_velocity, match_lanelet, 2)
                if min_heading_error > heading_error:
                    min_heading_error = heading_error
                    end_lanelet_index = index
            end_lanelet = obj_end_matches_rule_compliant[end_lanelet_index].lanelet

        return start_lanelet, end_lanelet

    def get_route_lanelet(self, start_lanelet, end_lanelet):
        lanelet_list = []
        if start_lanelet.id == end_lanelet.id:
            lanelet_list.append(start_lanelet)
        else:
            # print(start_lanelet.id, end_lanelet.id)
            lanelet_route = self._map.routing_graph.getRoute(start_lanelet, end_lanelet, 0)
            # print(lanelet_route)
            all_following_lanelet = lanelet_route.fullLane(start_lanelet)
            for lanelet in all_following_lanelet:
                lanelet_list.append(lanelet)
            if lanelet_list[0].id != start_lanelet.id:
                print('error route do not match start lanelet')
            if lanelet_list[-1].id != end_lanelet.id:
                print('error route do not match end lanelet')
                lanelet_list.append(end_lanelet)
        return lanelet_list

    def get_route_from_lanelet(self, route_lanelet, vehicle_start_pos, vehicle_end_pos):
        # we set the max speed of the vehicle as the recommand speed
        recommand_speed = 10 # m/s
        yaw_by_default = 0
        # all centerline points on the whole route
        centerline_point_list = []
        for lanelet in route_lanelet:
            if lanelet is route_lanelet[-1]:
                for index in range(len(lanelet.centerline)):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0]) # recommand_speed = sqrt(recommand_speed**2 + 0**2)
            else:
                for index in range(len(lanelet.centerline)-1):
                    centerline_point_list.append([lanelet.centerline[index].x, lanelet.centerline[index].y, yaw_by_default, recommand_speed, 0])

        # we just need a part of it
        condensed_centerline_point_list = []
        min_distance_with_start = 100
        min_distance_with_end = 100
        for index, point in enumerate(centerline_point_list):
            # find start centerline point's index
            distance_with_start = math.sqrt((point[0] - vehicle_start_pos[0])**2 + (point[1] - vehicle_start_pos[1])**2)
            if distance_with_start < min_distance_with_start:
                min_distance_with_start = distance_with_start
                start_index = index
            # find end centerline point's index
            distance_with_end = math.sqrt((point[0] - vehicle_end_pos[0])**2 + (point[1] - vehicle_end_pos[1])**2)
            if distance_with_end < min_distance_with_end:
                min_distance_with_end = distance_with_end
                end_index = index
        # make sure there are at least two points
        if start_index == end_index:
            end_index += 1

        for index in range(start_index, end_index + 1):
            condensed_centerline_point_list.append(centerline_point_list[index])
        
        # get route from the condensed centerline point list
        route = self.get_route_from_trajectory(trajectory_list=condensed_centerline_point_list)

        return route


    # generate ground truth routes
    def get_ground_truth_route(self, start_timestamp_list, interval_distance=2):
        ground_truth_route_dict = dict()
        track_dict = self._map.track_dict
        for vehicle_id in self._start_end_state.keys():
            vehicle_dict = track_dict[vehicle_id]
            # time horizen
            if len(start_timestamp_list) != 0:
                start_timestamp = int(start_timestamp_list[0])
                end_timestamp = start_timestamp + 100 * self._config.max_steps - 100
            else:
                start_timestamp = vehicle_dict.time_stamp_ms_first
                end_timestamp = vehicle_dict.time_stamp_ms_last
            ms_dict = vehicle_dict.motion_states
            vehicle_trajectory_list = self.get_trajectory_from_ms_dict(ms_dict, start_timestamp, end_timestamp)
            if vehicle_trajectory_list:
                ms_end = ms_dict[end_timestamp]
                vehicle_route_list = self.get_route_from_trajectory(trajectory_list=vehicle_trajectory_list, ms_end=ms_end)
                ground_truth_route_dict[vehicle_id] = vehicle_route_list

        return ground_truth_route_dict
        
    def get_route_from_trajectory(self, trajectory_list, interval_distance=2, ms_end=None):
        # a list [[x, y, point_recommend_speed]]

        # first make them equal distance
        average_trajectory_list = []
        for index, point in enumerate(trajectory_list):
            # first point
            if index == 0:
                average_trajectory_list.append([point[0], point[1]])
            # middle points
            elif index != (len(trajectory_list) - 1):
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                # distance is fine, just add it to the list
                if distance_to_previous >= 0.75 * interval_distance and distance_to_previous <= 1.25 * interval_distance:
                    average_trajectory_list.append([point[0], point[1]])
                # distace is too small, pass
                elif distance_to_previous < 0.75 * interval_distance:
                    continue
                # distance is too big, make it fine
                elif distance_to_previous > 1.25 * interval_distance:
                    ratio = 1.25 * interval_distance / distance_to_previous
                    insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                    insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                    average_trajectory_list.append([insert_point_x, insert_point_y])
            # last point
            else:
                point_previous = average_trajectory_list[-1]
                distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)
                if point[0:2] == point_previous:
                    if len(average_trajectory_list) > 1:
                        continue
                    else:
                        direction = ms_end.psi_rad
                        point_x =  point_previous[0] + interval_distance * math.cos(direction)
                        point_y =  point_previous[1] + interval_distance * math.sin(direction)
                        point = [point_x, point_y]
                        average_trajectory_list.append([point[0], point[1]])
                else:
                    # if distance too big, make it fine
                    while distance_to_previous > 1.25 * interval_distance:
                        ratio = 1.25 * interval_distance / distance_to_previous
                        insert_point_x = point_previous[0] + ratio * (point[0] - point_previous[0])
                        insert_point_y = point_previous[1] + ratio * (point[1] - point_previous[1])
                        average_trajectory_list.append([insert_point_x, insert_point_y])

                        point_previous = average_trajectory_list[-1]
                        distance_to_previous = math.sqrt((point[0] - point_previous[0])**2 + (point[1] - point_previous[1])**2)

                    average_trajectory_list.append([point[0], point[1]])

        # then the recommend speed value is the nearest trajectory point's speed value
        average_trajectory_with_speed_list = []
        for point in average_trajectory_list:
            min_distance = 100
            min_distance_point = None
            # get closest point in trajectory
            for point_with_speed in trajectory_list:
                distance = math.sqrt((point[0] - point_with_speed[0])**2 + (point[1] - point_with_speed[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_point = point_with_speed

            # calculate speed value
            point_speed = math.sqrt(min_distance_point[3] ** 2 + min_distance_point[4] ** 2)
            average_trajectory_with_speed_list.append([point[0], point[1], point_speed])

        return average_trajectory_with_speed_list

    def get_trajectory_from_ms_dict(self, ms_dict, start_timestamp, end_timestamp):
        # a list [[x, y, vehicle_yaw, vehicle_vx, vehicle_vy]...]
        trajectory_list = []
        # sort mc_dict based on time
        sorted_time = sorted(ms_dict)
        for time in sorted_time:
            if time >= start_timestamp and time <= end_timestamp:
                ms = ms_dict[time]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])
        # make sure the end point and start point's interval distance is long enough
        if trajectory_list: # if vehicle exist in the time horizen
            start_point = [trajectory_list[0][0], trajectory_list[0][1]]
            end_point = [trajectory_list[-1][0], trajectory_list[-1][1]]
            if math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2) < 2:
                ms = ms_dict[list(ms_dict.keys())[-1]]
                trajectory_list.append([ms.x, ms.y, ms.psi_rad, ms.vx, ms.vy])

        return trajectory_list

 

class sever_interface:
    def __init__(self, port):
        # communication related
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = port
        url = ':'.join(["tcp://*",str(self.port)])
        self.socket.bind(url)
        self.gym_env = None

        # env statue flag
        self.env_init_flag = False
        self.ego_choose_and_map_init_flag = False
        self.can_change_track_file_flag = False
        self.env_reset_flag = False
        print('server init')
                
    def start_communication(self):
        while not self.socket.closed:
            message = self.socket.recv()
            print('receive')
            str_message = bytes.decode(message)
            if str_message == 'close':
                self.socket.close()
                return
            print('decoded message:', message)
            message = eval(str_message)

            # env init
            if message['command'] == 'env_init':
                print('env init')
                self.gym_env = InteractionEnv(message['content'])
                self.socket.send_string('env_init_done')
                self.env_init_flag = True

            # choose ego & initialize map 
            elif message['command'] == 'ego_map_init':
                print('choose ego and initialize map')
                ego_id_list = self.gym_env.choose_ego_and_init_map(message['content'])
                self.socket.send_string(str(ego_id_list))
                self.ego_choose_and_map_init_flag = True

            # change track file
            elif message['command'] == 'track_init':
                print('choose track file')
                track_type = message['content']['track_type']
                track_content = message['content']['track_content']
                if track_type == 'predict':
                    self.gym_env.change_predict_track_file(trajectory_file_name=track_content)
                elif track_type == 'ground_truth':
                    self.gym_env.change_ground_truth_track_file(track_file_number=track_content)

                self.socket.send_string('change_file_done')
                self.can_change_track_file_flag = False

            # reset
            elif message['command'] == 'reset':
                print('reset')
                observation_dict = self.gym_env.reset()
                if observation_dict is not None:
                    self.env_reset_flag = True
                    # remove some unuseable item
                    condensed_observation_dict = self.pop_useless_item(observation_dict)
                    reset_message = {'observation': condensed_observation_dict, 'reward':0, 'done':False}
                    self.socket.send_string(str(reset_message).encode())
                else:
                    self.ego_choose_and_map_init_flag = False
                    self.socket.send_string(str(self.env_reset_flag).encode())

            # step
            elif message['command'] == 'step':
                action_dict = dict()
                # receiving action
                for ego_id in self.gym_env._ego_vehicle_dict.keys():
                    action_dict[ego_id] = message['content'][ego_id]

                observation_dict, reward_dict, done_dict, aux_info_dict = self.gym_env.step(action_dict)

                if False not in done_dict.values(): # all egos are done
                    self.can_change_track_file_flag = True
                    self.ego_choose_and_map_init_flag = False
                    self.env_reset_flag = False
                if observation_dict is not None:
                    condensed_observation_dict = self.pop_useless_item(observation_dict)
                    step_message = {'observation':condensed_observation_dict, 'reward': reward_dict, 'done': done_dict, 'aux_info': aux_info_dict}
                    self.socket.send_string(str(step_message).encode())

            else:
                print('env_init:', self.env_init_flag)
                print('ego_choose_and_map_init:', self.ego_choose_and_map_init_flag)
                print('can_change_track_file', self.can_change_track_file_flag)
                print('env_reset:', self.env_reset_flag)
                self.socket.send_string('null type')

    def pop_useless_item(self, observation):
        # remove some useless item from raw observation, to reduce communication costs
        useless_key = ['reach_end', 'collision', 'deflection', 'interaction_vehicles_id', 'future_route_points']
        observation_key = observation.keys()
        for item in useless_key:
            if item in observation_key:
                observation.pop(item)
        return observation

if __name__ == "__main__":

    # for docker external communication test
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Number of the port (int)", default=None, nargs="?")
    args = parser.parse_args()

    sever = sever_interface(args.port)
    sever.start_communication()
