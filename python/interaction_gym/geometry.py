import lanelet2
from lanelet2.core import AttributeMap, getId, BasicPoint2d, LineString3d, LineString2d, Point3d, Polygon2d
from lanelet2.geometry import inside, distance, intersects2d, length2d, intersectCenterlines2d, follows

import math
import numpy as np

def getAttributes():
    return AttributeMap({"key": "value"})

def is_equal_point(point1,point2):
    if point1.x == point2.x and point1.y == point2.y:
        return True
    else:
        return False

def lanelet_length(lanelet):
    # return centerline length
    length = length2d(lanelet)
    return length

def is_following_lanelet(previous_lanelet, next_lanelet):
    # check whether the next lanelet is the following lanelet of previous lanelet
    return follows(previous_lanelet,next_lanelet)

def insert_node_to_meet_min_interval(centerline_point_list, min_interval):
    # convert point form
    point_list = []
    if not isinstance(centerline_point_list[0], (list, tuple)):
        for point in centerline_point_list:
            point_list.append([point.x, point.y])
    else:
        for point in centerline_point_list:
            point_list.append([point[0], point[1]])        
    # uniform insert node to meet the minmum interval distance requirement
    extend_centerline_point_list = [] 
    for index in range(len(point_list)-1):
        extend_centerline_point_list.append(point_list[index])
        # print('origin point type:',type(centerline_point_list[index]))
        current_interval_distance =  math.sqrt((point_list[index][0] - point_list[index+1][0])**2 + (point_list[index][1] -point_list[index+1][1])**2)
        if current_interval_distance > min_interval:
            interval_num = math.ceil(current_interval_distance / min_interval)
            interval_point_num = interval_num - 1
            # print('interval_point_num:',interval_point_num)
            for i in range(int(interval_point_num)):
                pt_x = point_list[index][0] + (i+1) * (point_list[index+1][0] - point_list[index][0]) / interval_num
                pt_y = point_list[index][1] + (i+1) * (point_list[index+1][1] - point_list[index][1]) / interval_num
                # interval_point = Point3d(getId(), pt_x, pt_y, 0, getAttributes())
                interval_point = [pt_x, pt_y]
                extend_centerline_point_list.append(interval_point)
        
    extend_centerline_point_list.append(point_list[-1])

    return extend_centerline_point_list

def get_trajectory_distance(ego_pos, ego_trajectory_pos):
    trajectory_distance = math.sqrt((ego_pos[0] - ego_trajectory_pos[0]) ** 2 + (ego_pos[1] - ego_trajectory_pos[1]) ** 2)
    return trajectory_distance

def get_trajectory_pos(ego_state_dict, ego_trajectory_pos):
    ego_pos = ego_state_dict['pos']
    ego_heading = ego_state_dict['heading']
    x_in_ego_axis = (ego_trajectory_pos[1] - ego_pos[1])*np.cos(ego_heading) - (ego_trajectory_pos[0] - ego_pos[0])*np.sin(ego_heading)
    y_in_ego_axis = (ego_trajectory_pos[1] - ego_pos[1])*np.sin(ego_heading) + (ego_trajectory_pos[0] - ego_pos[0])*np.cos(ego_heading)
    return [x_in_ego_axis, y_in_ego_axis]

def get_trajectory_speed(ego_trajectory_velocity):
    # print('ego_trajectory_velocity:', ego_trajectory_velocity)
    trajectory_vx = ego_trajectory_velocity[0]
    trajectory_vy = ego_trajectory_velocity[1]

    trajectory_speed = math.sqrt(trajectory_vx**2 + trajectory_vy**2)

    return [trajectory_speed]

def ego_other_distance_and_collision(ego_state_dict, other_state_dict):
    # calculte the minmum distance between two polygon2d
    ego_polypoint_np = ego_state_dict['polygon']
    ego_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in ego_polypoint_np]
    ego_poly = Polygon2d(getId(),[ego_polyPoint3d[0],ego_polyPoint3d[1],ego_polyPoint3d[2],ego_polyPoint3d[3]],getAttributes())

    
    other_polypoint_np = other_state_dict['polygon']
    # print(type(v))
    other_polyPoint3d = [Point3d(getId(),p[0],p[1],0,getAttributes()) for p in other_polypoint_np]
    other_poly = Polygon2d(getId(),[other_polyPoint3d[0],other_polyPoint3d[1],other_polyPoint3d[2],other_polyPoint3d[3]],getAttributes())

    if intersects2d(ego_poly, other_poly):
        return 0,True
    else:   
        poly_distance = distance(ego_poly,other_poly)
        return poly_distance,False

def get_closet_bound_point(vehicle_pos, left_point_list, right_point_list):
    min_dist_l = 100
    min_dist_r = 100
    closet_left_point_index = 0
    closet_right_point_index = 0
    for index, point in enumerate(left_point_list):
        vehicle_to_point_dist = math.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)
        if min_dist_l > vehicle_to_point_dist:
            min_dist_l = vehicle_to_point_dist
            closet_left_point_index = index
    closet_left_point = [left_point_list[closet_left_point_index].x, left_point_list[closet_left_point_index].y]

    for index, point in enumerate(right_point_list):
        vehicle_to_point_dist = math.sqrt((point.x - vehicle_pos[0])**2 + (point.y - vehicle_pos[1])**2)
        if min_dist_r > vehicle_to_point_dist:
            min_dist_r = vehicle_to_point_dist
            closet_right_point_index = index
    closet_right_point = [right_point_list[closet_right_point_index].x, right_point_list[closet_right_point_index].y]

    road_width = math.sqrt((closet_right_point[0] - closet_left_point[0])**2 + (closet_right_point[1] - closet_left_point[1])**2)
    if min_dist_l > road_width:
        min_dist_r = 0
    elif  min_dist_r > road_width:
        min_dist_l = 0

    return [closet_left_point, closet_right_point], [min_dist_l, min_dist_r]


def get_vehicle_and_lanelet_heading_error(vehicle_pos, vehicle_heading, current_lanelet, min_interval_distance):
    # first find the closet centerline point
    # this function may need repair as the centerline points do not have uniform distance

    centerline_point_list = current_lanelet.centerline
    extend_centerline_point_list = insert_node_to_meet_min_interval(centerline_point_list, min_interval_distance)
    closet_point_index = get_closet_centerline_point(vehicle_pos, extend_centerline_point_list)

    # calculate the heading along the lanelet
    if closet_point_index < len(extend_centerline_point_list) - 1:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1].x - extend_centerline_point_list[closet_point_index].x, extend_centerline_point_list[closet_point_index+1].y - extend_centerline_point_list[closet_point_index].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index+1][0] - extend_centerline_point_list[closet_point_index][0], extend_centerline_point_list[closet_point_index+1][1] - extend_centerline_point_list[closet_point_index][1]])
    else:
        # lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index].x - extend_centerline_point_list[closet_point_index-1].x, extend_centerline_point_list[closet_point_index].y - extend_centerline_point_list[closet_point_index-1].y])
        lanelet_heading_vector = np.array([extend_centerline_point_list[closet_point_index][0] - extend_centerline_point_list[closet_point_index-1][0], extend_centerline_point_list[closet_point_index][1] - extend_centerline_point_list[closet_point_index-1][1]])

    vehicle_heading_vector = np.array([vehicle_heading[0],vehicle_heading[1]])

    L_lanelet_heading = np.sqrt(lanelet_heading_vector.dot(lanelet_heading_vector))
    # print('L_lanelet_heading:',L_lanelet_heading)
    L_vehicle_heading = np.sqrt(vehicle_heading_vector.dot(vehicle_heading_vector))
    # print('L_vehicle_heading:',L_vehicle_heading)
    cos_angle = vehicle_heading_vector.dot(lanelet_heading_vector)/(L_lanelet_heading*L_vehicle_heading)
    # print('cos_angle:',cos_angle)
    cos_angle = np.clip(cos_angle,-1,1)
    radian = np.arccos(cos_angle)
    heading_error =  radian * 180 / np.pi

    return heading_error

def get_route_bounds_points(route_lanelet, min_interval_distance):
    min_interval_distance = min_interval_distance/2
    left_bound_points = []
    right_bound_points = []
    for lanelet in route_lanelet:
        left_bound = lanelet.leftBound
        right_bound = lanelet.rightBound
        for i in range(len(left_bound)):
            left_bound_points.append(left_bound[i])
        for j in range(len(right_bound)):
            right_bound_points.append(right_bound[j])
    left_bound_points = insert_node_to_meet_min_interval(left_bound_points, min_interval_distance)
    right_bound_points = insert_node_to_meet_min_interval(right_bound_points, min_interval_distance)

    return left_bound_points, right_bound_points

def get_centerline_point_list_with_heading_and_average_interval(centerline_point_list, min_interval_distance):
    # calculate each point's heading
    previous_point_yaw = None
    centerline_point_list_with_heading = []

    for index, point in enumerate(centerline_point_list):
        if index == (len(centerline_point_list) - 1): # last centerlane point
            point_yaw = centerline_point_list_with_heading[-1][-1]
        else:
            point = centerline_point_list[index]
            point_next = centerline_point_list[index + 1]
            point_vector = np.array((point_next[0] - point[0], point_next[1] - point[1]))

            point_vector_length =  np.sqrt(point_vector.dot(point_vector))
            cos_angle = point_vector.dot(np.array(([1,0])))/(point_vector_length*1) # angle with x positive (same with carla)
            point_yaw = np.arccos(cos_angle) # rad
            if point_vector[1] < 0: # in the upper part of the axis, yaw is a positive value
                point_yaw = - point_yaw
            if previous_point_yaw:
                if (abs(point_yaw - previous_point_yaw) > np.pi/2 and abs(point_yaw - previous_point_yaw) < np.pi* (3/2)):
                    continue
                else:
                    previous_point_yaw = point_yaw
                   
            else:
                previous_point_yaw = point_yaw

        centerline_point_list_with_heading.append((point[0], point[1], point_yaw))

    return centerline_point_list_with_heading

def get_ego_route_point_with_heading_from_point_list(ego_predict_route, min_interval_distance):
    point_list = []
    point_previous = None
    for route_speedpoint in ego_predict_route:
        point_x = route_speedpoint[0]
        point_y = route_speedpoint[1]
        if [point_x, point_y] == point_previous:
            continue
        else:
            point_previous = [point_x, point_y]
        point_list.append([point_x, point_y])

    route_point_with_heading = get_centerline_point_list_with_heading_and_average_interval(point_list, min_interval_distance)

    return route_point_with_heading

def get_ego_target_speed_from_point_list(ego_predict_route):
    route_point_speed_list = []
    for route_point in ego_predict_route:
        point_speed = route_point[2]
        route_point_speed_list.append(point_speed)
    
    return route_point_speed_list

def get_closet_front_centerline_point(vehicle_pos, centerline_point_list_with_heading):
    min_dist = 100
    closet_point_index = 0
    for index, point in enumerate(centerline_point_list_with_heading):
        vehicle_to_point_dist = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
        vehicle_y_in_point_axis = (vehicle_pos[1] - point[1])*np.sin(point[2]) + (vehicle_pos[0] - point[0])*np.cos(point[2])
        if min_dist > vehicle_to_point_dist and vehicle_y_in_point_axis < 0:
            min_dist = vehicle_to_point_dist
            closet_point_index = index

    return closet_point_index

def get_lane_observation_and_future_route_points(ego_state_dict, vehilce_route, vehilce_route_target_speed, control_steering, normalization):
    vehicle_pos = ego_state_dict['pos']
    vehicle_heading = ego_state_dict['heading']
    vehicle_speed = ego_state_dict['speed']
    
    future_points = [] # for render next 5 route points
    # first find the closet route point
    closet_point_index = get_closet_front_centerline_point(vehicle_pos, vehilce_route)

    center_point = vehilce_route[closet_point_index]
    current_target_speed = vehilce_route_target_speed[closet_point_index]

    future_points.append(vehilce_route[closet_point_index])
    print('current_lane_heading', center_point[2] * 180 / np.pi)
    # print(vehicle_pos, center_point)
    
    ego_x_in_point_axis = (vehicle_pos[1] - center_point[1])*np.cos(center_point[2]) - (vehicle_pos[0] - center_point[0])*np.sin(center_point[2])
    ego_y_in_point_axis = (vehicle_pos[1] - center_point[1])*np.sin(center_point[2]) + (vehicle_pos[0] - center_point[0])*np.cos(center_point[2])

    ego_heading_error_0 = center_point[2] - vehicle_heading
    ego_speed_x_in_point_axis = vehicle_speed * np.sin(ego_heading_error_0)
    ego_speed_y_in_point_axis = vehicle_speed * np.cos(ego_heading_error_0)

    # also get next 4 points' heading errors
    ego_heading_error_next_list = []
    require_num = 4
    remain_point_num = len(vehilce_route) - 1 - closet_point_index 
    if remain_point_num < require_num:
        for i in range(closet_point_index + 1, len(vehilce_route)):
            point_heading = vehilce_route[i][2]
            ego_heading_error_point = point_heading - vehicle_heading
            ego_heading_error_next_list.append(ego_heading_error_point)
            future_points.append(vehilce_route[i])
        while len(ego_heading_error_next_list) < require_num:
            if ego_heading_error_next_list:
                ego_heading_error_next_list.append(ego_heading_error_next_list[-1])
            else:
                ego_heading_error_next_list.append(ego_heading_error_0)
    else:
        for i in range(closet_point_index + 1, closet_point_index + require_num + 1):
            point_heading = vehilce_route[i][2]
            ego_heading_error_point = point_heading - vehicle_heading
            ego_heading_error_next_list.append(ego_heading_error_point)
            future_points.append(vehilce_route[i])

    if control_steering:
        if normalization:
            ego_x_in_point_axis /= 2
            ego_speed_x_in_point_axis /= 10
            ego_speed_y_in_point_axis /= 10
            ego_heading_error_0 /= np.pi
            ego_heading_error_next_list = [i/np.pi for i in ego_heading_error_next_list]

        lane_observation = [ego_x_in_point_axis, ego_speed_x_in_point_axis, ego_speed_y_in_point_axis, ego_heading_error_0] + ego_heading_error_next_list
    else:
        lane_observation = [ego_heading_error_0] + ego_heading_error_next_list

    return lane_observation, current_target_speed, future_points, ego_y_in_point_axis

def get_ego_next_pos(ego_state_dict):
    next_point_pos_x = ego_state_dict['pos'][0] + ego_state_dict['speed'] * math.cos(ego_state_dict['heading'])
    next_point_pos_y = ego_state_dict['pos'][1] + ego_state_dict['speed'] * math.sin(ego_state_dict['heading'])
    next_point_pos = (next_point_pos_x, next_point_pos_y)

    next_x_in_ego_axis = (next_point_pos[1] - ego_state_dict['pos'][1])*np.cos(ego_state_dict['heading']) - (next_point_pos[0] - ego_state_dict['pos'][0])*np.sin(ego_state_dict['heading'])
    next_y_in_ego_axis = (next_point_pos[1] - ego_state_dict['pos'][1])*np.sin(ego_state_dict['heading']) + (next_point_pos[0] - ego_state_dict['pos'][0])*np.cos(ego_state_dict['heading'])
    # print('next_pos_from_ego', [next_x_in_ego_axis, next_y_in_ego_axis])
    return [next_x_in_ego_axis, next_y_in_ego_axis]
