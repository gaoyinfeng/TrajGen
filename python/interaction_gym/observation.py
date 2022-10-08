#-*- coding: UTF-8 -*- 
import math
import matplotlib
import numpy as np
import heapq
try:
    import lanelet2
    import lanelet2_matching
    print("Lanelet2_matching import")
except:
    import warnings
    string = "Could not import lanelet2_matching"
    warnings.warn(string)
import geometry

class Observation:
    def __init__(self, ego_vehicles_dict, interaction_map, config, control_steering, normalization=False):
        self._map = interaction_map
        self._ego_vehicles_dict = ego_vehicles_dict

        self._config = config
        self._control_steering = control_steering

        # ego routes related info
        self.ego_route_dict = dict()
        self.ego_closet_bound_points = dict()

        # termination judgement
        self.reach_goal = dict()
        self.collision = dict()
        self.deflection = dict()

        # observation terms
        self.trajectory_distance = dict()
        self.trajectory_pos = dict()
        self.trajectory_speed = dict()
        
        self.distance_from_bound = dict()
        self.lane_observation = dict()

        self.future_route_points = dict()

        self.ego_shape = dict()
        self.ego_route_points = dict()
        self.ego_route_target_speed = dict()
        self.ego_route_left_bound_points = dict()
        self.ego_route_right_bound_points = dict()
        self.ego_current_speed = dict()
        self.ego_current_target_speed = dict()
        self.ego_next_pos = dict()

        self.interaction_vehicles_id = dict()
        self.interaction_vehicles_observation = dict()
        self.attention_mask = dict()

        self.observation_dict = dict()

        # init
        for ego_id in self._ego_vehicles_dict.keys():
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

        self.virtual_route_bound = True
        self.normalization = normalization

        self.min_interval_distance = 2 # minmum interval distance of waypoint (meter)
        

    def reset(self, route_type, route_dict):
        self.route_type = route_type
        self.ego_route_dict = route_dict

        for ego_id in self._ego_vehicles_dict.keys():
            # self.reach_end_lanelet[k] = False
            self.reach_goal[ego_id] = False
            self.collision[ego_id] = False
            self.deflection[ego_id] = False

        self.trajectory_distance.clear()
        self.trajectory_pos.clear()
        self.trajectory_speed.clear()

        self.ego_shape.clear()
        self.ego_route_points.clear()
        self.ego_route_target_speed.clear()
        self.ego_route_left_bound_points.clear()
        self.ego_route_right_bound_points.clear()
        self.ego_closet_bound_points.clear()
        self.distance_from_bound.clear()
        self.lane_observation.clear()
        self.future_route_points.clear()
        self.ego_next_pos.clear()

        self.ego_current_speed.clear()
        self.ego_current_target_speed.clear()
        self.interaction_vehicles_id.clear()
        self.interaction_vehicles_observation.clear()
        self.attention_mask.clear()

        self.observation_dict.clear()

        return True

    def get_interaction_vehicles_id_and_observation(self, ego_state_dict, other_vehicles_state_dict):
        ego_pos = ego_state_dict['pos']
        ego_heading = ego_state_dict['heading']

        surrounding_vehicles = []
        # 1. check if this vehicle within ego's detective range, and put them together
        ego_detective_range = 30 # m
        for other_id, other_state_dict in other_vehicles_state_dict.items():
            # motion state
            other_vehicle_pos = other_state_dict['pos']
            other_vehicle_speed = other_state_dict['speed']
            other_vehicle_heading = other_state_dict['heading']

            distance_with_ego = math.sqrt((other_vehicle_pos[0] - ego_pos[0])**2 + (other_vehicle_pos[1] - ego_pos[1])**2)
            y_relative = (other_vehicle_pos[1] - ego_pos[1])*np.sin(ego_heading) + (other_vehicle_pos[0] - ego_pos[0])*np.cos(ego_heading)
            if distance_with_ego <= ego_detective_range and y_relative > -12:
                add_dict = {'vehicle_id': other_id, 'distance': distance_with_ego, 'pos': other_vehicle_pos, 'speed': other_vehicle_speed, 'heading': other_vehicle_heading}
                surrounding_vehicles.append(add_dict)

        # 2. get interaction vehicles and their basic observation
        interaction_vehicles = heapq.nsmallest(self._config.npc_num, surrounding_vehicles, key=lambda s: s['distance'])

        # 3. get their ids and full observation
        interaction_vehicles_id = []
        interaction_vehicles_observation = []
        for vehicle_dict in interaction_vehicles:
            # id
            interaction_vehicles_id.append(vehicle_dict['vehicle_id'])
            # basic observation
            # shape
            other_vehicle_polygan = other_vehicles_state_dict[vehicle_dict['vehicle_id']]['polygon']
            poly_01 = [i - j for i, j in zip(other_vehicle_polygan[0], other_vehicle_polygan[1])]
            poly_12 = [i - j for i, j in zip(other_vehicle_polygan[1], other_vehicle_polygan[2])]
            vehicle_length = math.sqrt(poly_01[0]**2 + poly_01[1]**2)
            vehicle_width = math.sqrt(poly_12[0]**2 + poly_12[1]**2)
            # motion state
            vehicle_pos = vehicle_dict['pos']
            vehicle_speed = vehicle_dict['speed']
            vehicle_heading = vehicle_dict['heading']

            # ture observation
            x_in_ego_axis = (vehicle_pos[1] - ego_pos[1])*np.cos(ego_heading) - (vehicle_pos[0] - ego_pos[0])*np.sin(ego_heading)
            y_in_ego_axis = (vehicle_pos[1] - ego_pos[1])*np.sin(ego_heading) + (vehicle_pos[0] - ego_pos[0])*np.cos(ego_heading)
            heading_error_with_ego = vehicle_heading - ego_heading

            single_observation = [vehicle_length, vehicle_width, x_in_ego_axis, y_in_ego_axis, vehicle_speed, np.cos(heading_error_with_ego), np.sin(heading_error_with_ego)]
            interaction_vehicles_observation += single_observation
        
        # 4. zero padding and attention mask
        attention_mask = list(np.ones(self._config.mask_num))
        npc_obs_size = self._config.npc_num * self._config.npc_feature_num
        if len(interaction_vehicles_observation) < npc_obs_size:
            zero_padding_num = int( (npc_obs_size - len(interaction_vehicles_observation)) / self._config.npc_feature_num)
            for _ in range(zero_padding_num):
                attention_mask.pop()
            for _ in range(zero_padding_num):
                attention_mask.append(0)
            while len(interaction_vehicles_observation) < npc_obs_size:
                interaction_vehicles_observation.append(0)

        return interaction_vehicles_id, interaction_vehicles_observation, attention_mask

    def get_intersection_vehicle_id(self, observation_dict):
        intersection_vehicle_id = []
        for ego_id in self._ego_vehicles_dict.keys():
            intersection_vehicle_id += observation_dict['interaction_vehicles_id'][ego_id]

        return intersection_vehicle_id

    def get_future_route_points(self, observation_dict):
        future_route_points_dict = dict()
        for ego_id in self._ego_vehicles_dict.keys():
            future_route_points_dict[ego_id] = observation_dict['future_route_points'][ego_id]

        return future_route_points_dict

    def get_current_bound_points(self, observation_dict):
        current_bound_points = []
        for ego_id in self._ego_vehicles_dict.keys():
            current_bound_points += observation_dict['current_bound_points'][ego_id]

        return current_bound_points

    def check_ego_reach_goal(self, ego_state_dict, goal_point):
        ego_loc_x = ego_state_dict['pos'][0]
        ego_loc_y = ego_state_dict['pos'][1]

        goal_loc_x = goal_point[0]
        goal_loc_y = goal_point[1]

        ego_goal_distance = math.sqrt((ego_loc_x - goal_loc_x)**2 + (ego_loc_y - goal_loc_y)**2)
        return ego_goal_distance < 2

    def check_ego_collision(self, ego_state_dict, other_vehicles_state_dict, interaction_vehicles_id):
        ego_collision = False

        for other_id, other_state_dict in other_vehicles_state_dict.items():
            if other_id in interaction_vehicles_id:
                distance, collision = geometry.ego_other_distance_and_collision(ego_state_dict, other_state_dict)
                if collision:
                    return True
        return False

    def check_ego_deflection(self, virtual_route_bound, limitation, distance_bound=None, distance_to_center=None, ego_y_in_point_axis=None):
        deflection = False
        if virtual_route_bound:
            if distance_to_center > limitation or ego_y_in_point_axis > 0:
                deflection = True
        else:
            if distance_bound < limitation:
                deflection = True
        return deflection


    def get_scalar_observation(self, current_time):
        for ego_id, ego_state in self._ego_vehicles_dict.items():
            # get ego shape, polygon and motion states
            ego_state_dict = dict()
            self.ego_shape[ego_id] = [ego_state._length, ego_state._width]
            ego_state_dict['polygon'] = self._map.ego_vehicle_polygon[ego_id]
            ego_state_dict['pos'] = [ego_state._current_state.x, ego_state._current_state.y]
            ego_state_dict['speed'] = math.sqrt(ego_state._current_state.vx ** 2 + ego_state._current_state.vy ** 2)
            ego_state_dict['heading'] = ego_state._current_state.psi_rad
            
            # get other vehicless state, first other egos, then the log npcs
            other_vehicles_state_dict = dict()
            for other_ego_id, other_ego_state in self._ego_vehicles_dict.items():
                if other_ego_id == ego_id:
                    continue
                else:
                    other_vehicles_state_dict[other_ego_id] = dict()
                    other_vehicles_state_dict[other_ego_id]['polygon'] = self._map.ego_vehicle_polygon[other_ego_id]
                    other_vehicles_state_dict[other_ego_id]['pos'] = [other_ego_state._current_state.x, other_ego_state._current_state.y]
                    other_vehicles_state_dict[other_ego_id]['speed'] = math.sqrt(other_ego_state._current_state.vx ** 2 + other_ego_state._current_state.vy ** 2)
                    other_vehicles_state_dict[other_ego_id]['heading'] = other_ego_state._current_state.psi_rad
            for other_npc_id, other_npc_polygon in self._map.other_vehicle_polygon.items():
                other_vehicles_state_dict[other_npc_id] = dict()
                other_npc_motion_state = self._map.other_vehicle_motion_state[other_npc_id]
                other_vehicles_state_dict[other_npc_id]['pos'] = [other_npc_motion_state.x, other_npc_motion_state.y]
                other_vehicles_state_dict[other_npc_id]['speed'] = math.sqrt(other_npc_motion_state.vx ** 2 + other_npc_motion_state.vy ** 2)
                other_vehicles_state_dict[other_npc_id]['heading'] = other_npc_motion_state.psi_rad
                other_vehicles_state_dict[other_npc_id]['polygon'] = other_npc_polygon

            # get current ego route point
            if self.route_type == 'predict':
                self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id])
                # do not need lane bound and distance if use predict route (for now)
                # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = None, None
                # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = None, None

            elif self.route_type == 'ground_truth':
                self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id])
                # get current lane bound and distance
                # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(route_lanelet, self.min_interval_distance)
                # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['pos'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])
            
            elif self.route_type == 'centerline':
                # self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_lanelet(route_lanelet, self.min_interval_distance)
                self.ego_route_points[ego_id] = geometry.get_ego_route_point_with_heading_from_point_list(self.ego_route_dict[ego_id], self.min_interval_distance)
                self.ego_route_target_speed[ego_id] = geometry.get_ego_target_speed_from_point_list(self.ego_route_dict[ego_id]) # we set the max speed of the vehicle as the target speed
                # get current lane bound and distance
                # self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id] = geometry.get_route_bounds_points(route_lanelet, self.min_interval_distance)
                # self.ego_closet_bound_points[ego_id], self.distance_from_bound[ego_id] = geometry.get_closet_bound_point(ego_state_dict['pos'], self.ego_route_left_bound_points[ego_id], self.ego_route_right_bound_points[ego_id])

            # get ego's distance with ground truth trajectory (for ade and fde calculation)
            ego_trajectory = self._map._ego_vehicle_track_dict[ego_id]
            ego_trajectory_pos = [ego_trajectory.motion_states[current_time].x, ego_trajectory.motion_states[current_time].y]
            ego_trajectory_velocity = [ego_trajectory.motion_states[current_time].vx, ego_trajectory.motion_states[current_time].vy] 
            trajectory_distance = geometry.get_trajectory_distance(ego_state_dict['pos'], ego_trajectory_pos)
            self.trajectory_distance[ego_id] = [trajectory_distance]
            trajectory_pos = geometry.get_trajectory_pos(ego_state_dict, ego_trajectory_pos)
            self.trajectory_pos[ego_id] = trajectory_pos
            trajectory_speed = geometry.get_trajectory_speed(ego_trajectory_velocity)
            self.trajectory_speed[ego_id] = trajectory_speed

            # ego current speed value
            self.ego_current_speed[ego_id] = [ego_state_dict['speed']]

            # ego distance, heading errors and velocity from route center
            lane_observation, ego_current_target_speed, future_route_points, ego_y_in_point_axis = geometry.get_lane_observation_and_future_route_points(ego_state_dict, self.ego_route_points[ego_id], self.ego_route_target_speed[ego_id], self._control_steering, self.normalization)
            self.lane_observation[ego_id] = lane_observation
            self.ego_current_target_speed[ego_id] = [ego_current_target_speed]
            self.future_route_points[ego_id] = future_route_points
            
            # ego's next position raletive to current
            self.ego_next_pos[ego_id] = geometry.get_ego_next_pos(ego_state_dict)

            # get interaction social vehicles' id and observation
            interaction_vehicles_id, interaction_vehicles_observation, attention_mask = self.get_interaction_vehicles_id_and_observation(ego_state_dict, other_vehicles_state_dict)
            self.interaction_vehicles_id[ego_id] = interaction_vehicles_id
            self.interaction_vehicles_observation[ego_id] = interaction_vehicles_observation
            self.attention_mask[ego_id] = attention_mask

            # Finish judgement 1: reach goal
            goal_point = self.ego_route_points[ego_id][-1]
            reach_goal = self.check_ego_reach_goal(ego_state_dict, goal_point)
            self.reach_goal[ego_id] = reach_goal
            
            # Finish judgement 2: collision with other vehicles
            ego_collision = self.check_ego_collision(ego_state_dict, other_vehicles_state_dict, interaction_vehicles_id)
            self.collision[ego_id] = ego_collision

            # Finish judgement 3: deflection from current route/road
            if self._control_steering:
                if self.virtual_route_bound:
                    ego_x_in_route_axis = self.lane_observation[ego_id][0]
                    limitation = 3
                    ego_deflection = self.check_ego_deflection(virtual_route_bound=True, limitation=limitation, distance_to_center=abs(ego_x_in_route_axis), ego_y_in_point_axis=ego_y_in_point_axis)
                else: # actual route bound
                    ego_min_bound_distance = min(self.distance_from_bound[ego_id])
                    limitation = 0.25
                    ego_deflection = self.check_ego_deflection(virtual_route_bound=False, limitation=limitation, distance_bound=ego_min_bound_distance)
            else:
                ego_deflection = False
            self.deflection[ego_id] = ego_deflection

        # Finish judgements
        self.observation_dict['reach_end'] = self.reach_goal
        self.observation_dict['collision'] = self.collision
        self.observation_dict['deflection'] = self.deflection

        # Observations - ego state
        self.observation_dict['ego_shape'] = self.ego_shape # 2-D
        self.observation_dict['current_speed'] = self.ego_current_speed      # 1-D
        self.observation_dict['ego_next_pos'] = self.ego_next_pos  # 2-D

        # Observations - others state
        self.observation_dict['interaction_vehicles_observation'] = self.interaction_vehicles_observation  # 35-D
        
        # Observations - route tracking
        self.observation_dict['trajectory_pos'] = self.trajectory_pos              # 2-D
        self.observation_dict['trajectory_speed'] = self.trajectory_speed          # 1-D
        self.observation_dict['trajectory_distance'] = self.trajectory_distance    # 1-D
        self.observation_dict['target_speed'] = self.ego_current_target_speed      # 1-D
        self.observation_dict['distance_from_bound'] = self.distance_from_bound    # 2-D
        self.observation_dict['lane_observation'] = self.lane_observation # 8-D
        
        # Observations - attention mask
        self.observation_dict['attention_mask'] = self.attention_mask # 6-D

        # Observations - render
        self.observation_dict['interaction_vehicles_id'] = self.interaction_vehicles_id  # use for render
        # self.observation_dict['current_bound_points'] = self.ego_closet_bound_points     # use for render
        self.observation_dict['future_route_points'] = self.future_route_points  # use for render
        
        return self.observation_dict

    def get_ego_center_image(self, current_time, ego_state_dict):
        image_array = np.asarray(self._map.fig.canvas.buffer_rgba())
        # print(image_array.shape)
        image_array = image_array.reshape(self._map.fig_height, self._map.fig_width, 4)
        for k,v in ego_state_dict.items():
            ego_center_pixel_x = int((v.x - self._map.map_x_bound[0]) / self._map.map_width_ratio)
            ego_center_pixel_y = int((v.y - self._map.map_y_bound[0]) / self._map.map_height_ratio)
            image_min_x = ego_center_pixel_x - self._map.image_half_width
            image_max_x = ego_center_pixel_x + self._map.image_half_width
            image_min_y = ego_center_pixel_y - self._map.image_half_height
            image_max_y = ego_center_pixel_y + self._map.image_half_height
            ego_center_image = image_array[self._map.fig_height - image_max_y:self._map.fig_height- image_min_y,image_min_x:image_max_x,:].tolist()
            # matplotlib.image.imsave('ego_'+ str(k) + '_' + str(current_time) +'.png', ego_center_image)
            matplotlib.image.imsave('ego_'+ str(k) + '_' + str(current_time) +'.png', image_array.tolist())
        # matplotlib.image.imsave('map.png', image_array.tolist())

    def get_visualization_observation(self, current_time, ego_state_dict):
        self.get_ego_center_image(current_time, ego_state_dict)