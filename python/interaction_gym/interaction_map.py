
try:
    import lanelet2
    print("Using Lanelet2 visualization")
except:
    import warnings
    string = "Could not import lanelet2. It must be built and sourced, " + \
             "see https://github.com/fzi-forschungszentrum-informatik/Lanelet2 for details."
    warnings.warn(string)

import random
import argparse
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import geometry

from utils.dataset_reader import read_trajectory, read_tracks, read_pedestrian
from utils.visualize import update_ego_others_param, render_vehicles_with_highlight
from utils.map_vis_lanelet2 import draw_lanelet_map, draw_route, draw_ego_future_route, draw_route_bounds, draw_closet_bound_point

# the map data in intersection dataset
# x axis direction is from left to right
# y axis direction is from top to bottom

class InteractionMap:
    def __init__(self, args):
        # data dir path
        self._root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._map_dir = os.path.join(self._root_dir, "maps")
        self._tracks_dir = os.path.join(self._root_dir, "recorded_trackfiles")
        lanelet_map_ending = ".osm"

        self._lanelet_map_file = os.path.join(self._map_dir, args['scenario_name'] + lanelet_map_ending)
        self._scenario_dir = os.path.join(self._tracks_dir, args['scenario_name'])

        # check folders and files
        error_string = ""
        if not os.path.isdir(self._tracks_dir):
            error_string += "Did not find track file directory \"" + self._tracks_dir + "\"\n"
        if not os.path.isdir(self._map_dir):
            error_string += "Did not find map file directory \"" + self._tracks_dir + "\"\n"
        if not os.path.isdir(self._scenario_dir):
            error_string += "Did not find scenario directory \"" + self._scenario_dir + "\"\n"
        if not os.path.isfile(self._lanelet_map_file):
            error_string += "Did not find lanelet map file \"" + self._lanelet_map_file + "\"\n"
        else:
            flag_ped = 1
        if error_string != "":
            error_string += "Type --help for help."
            raise IOError(error_string)

        # load pedestrian track if needed
        self._pedestrian_dict = dict()
        if args['load_mode'] == 'both' or args['load_mode'] == 'pedestrian' and flag_ped:
            self._pedestrian_dict = read_pedestrian(self._pedestrian_file_name)
        
        # create a figure
        self._fig, self._axes = plt.subplots(1, 1, facecolor = 'lightgray')  # figure backcolor (figure size > map render size)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

        self._axes.set_facecolor('white') # map render backcolor
        self._fig.canvas.set_window_title("Interaction Env Visualization " + str(args['port']))

        # load and draw the lanelet2 map
        lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
        print("Loading map...")
        
        self._laneletmap = None
        self._rules_map = {"vehicle": lanelet2.traffic_rules.Participants.Vehicle}
        self._projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))


        # initialize vehicles' id list and so on
        self._ego_vehicle_id_list = list()
        self._others_vehicle_id_list = list()

        self._ego_vehicle_track_dict = dict()
        self._other_vehicle_track_dict = dict()

        self._ego_vehicle_start_end_state_dict = dict()

        self._ego_patches_dict = dict()
        self._other_patches_dict = dict()
        self._ghost_patches_dict = dict()

        # use for collison detection and distance calculation
        self.ego_vehicle_polygon = dict()
        self.other_vehicle_polygon = dict()
        self._ghost_vehicle_polygon = dict()

        self.other_vehicle_motion_state = dict()
        self._ghost_vehicle_motions_state = dict()

        # use for vehicle id visualaztion
        self._text_dict = dict()

        self.track_dict = dict()
                

    def __del__(self):
        plt.close('all')


    def change_predict_track_file(self, trajectory_file_name=None):
        self._trajectory_file_name = trajectory_file_name
        self.track_dict = read_trajectory(self._trajectory_file_name)
        print('predict track file name is:', trajectory_file_name)
    

    def change_ground_truth_track_file(self, track_file_number=None, trajectory_file_name=None):
        self._track_file_name = os.path.join(
            self._scenario_dir,
            "vehicle_tracks_" + str(track_file_number).zfill(3) + ".csv")
        self.track_dict = read_tracks(self._track_file_name)
        print('ground truth track file number is:', track_file_number)


    def map_init(self):
        # clear
        self._ego_vehicle_start_end_state_dict.clear()

        self._ego_vehicle_track_dict.clear()
        self._other_vehicle_track_dict.clear()

        self._ego_patches_dict.clear()
        self._other_patches_dict.clear()
        self._ghost_patches_dict.clear()

        self.ego_vehicle_polygon.clear()
        self.other_vehicle_polygon.clear()
        self._ghost_vehicle_polygon.clear()

        self.other_vehicle_motion_state.clear()
        self._ghost_vehicle_motions_state.clear()

        self._axes.clear()

        # initialize map
        self._laneletmap = lanelet2.io.load(self._lanelet_map_file, self._projector)

        # render static map and get pixel ratio
        self._map_x_bound, self._map_y_bound = draw_lanelet_map(self._laneletmap, self._axes)
        self._fig_width, self._fig_height = self._fig.get_size_inches() * self._fig.dpi
        self._fig_width = int(self._fig_width) # figure size >= map render size as there exist empty space between figure bound and axes bound
        self._fig_height = int(self._fig_height)
        # pixel ration: (m/pixel)
        self._map_width_ratio = (self._map_x_bound[1] - self._map_x_bound[0])/self._fig_width
        self._map_height_ratio = (self._map_y_bound[1] - self._map_y_bound[0])/self._fig_height

        # ego fixed pos image size 
        self._image_half_width = int(15 / self._map_width_ratio)
        self._image_half_height = int(15 / self._map_height_ratio)
        
        self._routing_cost = lanelet2.routing.RoutingCostDistance(0.) # zero cost for lane changes
        self._traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, self._rules_map['vehicle'])
        
        # route graph is used for lanelet relationship searching
        self.routing_graph = lanelet2.routing.RoutingGraph(self._laneletmap, self._traffic_rules, [self._routing_cost])

    
    def random_choose_ego_vehicle(self, max_length=5.5):
        # randome select vehicles whose length is less than max_length as egos
        vehicle_id_list = self.track_dict.keys()
        vehicle_limited_length_id_list = []
        for vehicle_id in vehicle_id_list:
            vehicle_info = self.track_dict[vehicle_id]
            if vehicle_info.length <= max_length:
                vehicle_limited_length_id_list.append(vehicle_id)

        # split track into egos' and others'
        self._ego_vehicle_id_list = random.sample(vehicle_limited_length_id_list, self._ego_vehicle_num)
        self._others_vehicle_id_list = list(set(vehicle_id_list) - set(self._ego_vehicle_id_list))

        self._ego_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._ego_vehicle_id_list}
        self._other_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._others_vehicle_id_list}

        # read ego vehicles start end state for env reset
        for ego_id, vehicle_id in enumerate(self._ego_vehicle_id_list):
            ego_info = self.track_dict[vehicle_id]
            length = ego_info.length
            width = ego_info.width

            ego_timestamp_ms_first = ego_info.time_stamp_ms_first
            ego_timestamp_ms_last = ego_info.time_stamp_ms_last
            self._ego_vehicle_start_end_state_dict[vehicle_id] = [ego_timestamp_ms_first, ego_timestamp_ms_last, length, width, ego_info.motion_states[ego_timestamp_ms_first], ego_info.motion_states[ego_timestamp_ms_last]]
            
    def specify_id_choose_ego_vehicle(self, ego_id_list, ego_start_timestamp=None):
        # split track into egos' and others'
        vehicle_id_list = self.track_dict.keys()
        self._ego_vehicle_id_list = ego_id_list
        self._others_vehicle_id_list = list(set(vehicle_id_list) - set(self._ego_vehicle_id_list))
        
        self._ego_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._ego_vehicle_id_list}
        self._other_vehicle_track_dict = {key:value for key,value in self.track_dict.items() if key in self._others_vehicle_id_list}

        # read ego vehicles start & end state
        for ego_id, vehicle_id in enumerate(self._ego_vehicle_id_list):
            ego_info = self.track_dict[vehicle_id]
            length = ego_info.length
            width = ego_info.width

            if ego_start_timestamp:
                ego_timestamp_ms_first = int(ego_start_timestamp[0])
                ego_timestamp_ms_last = ego_timestamp_ms_first + 99 * 100
            else:
                ego_timestamp_ms_first = ego_info.time_stamp_ms_first
                ego_timestamp_ms_last = ego_info.time_stamp_ms_last

            self._ego_vehicle_start_end_state_dict[vehicle_id] = [ego_timestamp_ms_first, ego_timestamp_ms_last, length, width, ego_info.motion_states[ego_timestamp_ms_first], ego_info.motion_states[ego_timestamp_ms_last]]


    def update_param(self, current_time, ego_state_dict):
        # visualize map and vehicles
        ego_shape_dict = dict()
        for vehicle_id, vehicle_info in self._ego_vehicle_start_end_state_dict.items():
            ego_shape_dict[vehicle_id] = (vehicle_info[2], vehicle_info[3]) # length and width
        # pack different info
        ego_info = {"shape": ego_shape_dict,
                    "state": ego_state_dict,
                    "polygon": self.ego_vehicle_polygon,
                    "track": self._ego_vehicle_track_dict
                    }
        other_info = {"state": self.other_vehicle_motion_state,
                    "polygon": self.other_vehicle_polygon,
                    "track": self._other_vehicle_track_dict
                    }
        ghost_info = {"state": self._ghost_vehicle_motions_state,
                    "polygon": self._ghost_vehicle_polygon,
                    }
        # update ego vehicles and other vehicles
        update_ego_others_param(current_time, ego_info, other_info, ghost_info, self._pedestrian_dict)
     
    # render ego and other vehicles
    def render_vehicles(self, ego_state_dict, highlight_vehicle_id_list, ghost_vis=True):
        plt.ion()

        # pack different info
        ego_info = {"state": ego_state_dict,
                    "polygon": self.ego_vehicle_polygon,
                    "track": self._ego_vehicle_track_dict,
                    "patch": self._ego_patches_dict
                    }
        other_info = {"state": self.other_vehicle_motion_state,
                    "polygon": self.other_vehicle_polygon,
                    "track": self._other_vehicle_track_dict,
                    "patch": self._other_patches_dict
                    }
        ghost_info = {"state": self._ghost_vehicle_motions_state,
                    "polygon": self._ghost_vehicle_polygon,
                    'patch': self._ghost_patches_dict
                    }

        render_vehicles_with_highlight(ego_info, other_info, ghost_info, highlight_vehicle_id_list, self._text_dict, self._axes, self._fig, ghost_vis)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.pause(0.001)
        plt.show()
        plt.ioff()

    # render routes
    def render_route(self, route):
        plt.ion()

        for k,v in route.items():
            route_point_list = []
            for point in v:
                route_point_list.append([point[0], point[1]])
            draw_route(route_point_list, self._axes)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()

    def render_future_route_points(self, ego_previous_points_dict, ego_future_points_dict):
        
        plt.ion()

        draw_ego_future_route(ego_previous_points_dict, ego_future_points_dict, self._axes)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()

    # render route bounds
    def render_route_bounds(self, route_left_bounds, route_right_bounds):
        plt.ion()

        for k,v in route_left_bounds.items():
            draw_route_bounds(route_left_bounds[k], self._axes)
            draw_route_bounds(route_right_bounds[k], self._axes)

        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()

    def render_closet_bound_point(self, previous_closet_points, current_closet_points):
        plt.ion()

        draw_closet_bound_point(previous_closet_points, current_closet_points, self._axes)
        
        plt.xticks([]) 
        plt.yticks([])
        plt.axis('off')

        plt.show()
        plt.ioff()
