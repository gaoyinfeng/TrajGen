import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from matplotlib.collections import PatchCollection
import math

from interaction_gym import geometry


def set_get_visible_area(laneletmap, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for point in laneletmap.pointLayer:
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)
    
    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])
    # x_lim() return same value as get_xbound()
    map_x_bound = axes.get_xbound()
    map_y_bound = axes.get_ybound()
    
    return map_x_bound,map_y_bound


def draw_lanelet_map(laneletmap, axes):

    assert isinstance(axes, matplotlib.axes.Axes)

    map_x_bound,map_y_bound = set_get_visible_area(laneletmap, axes)

    unknown_linestring_types = list()

    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            # type_dict = dict(color="white", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                # type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
                continue
            else:
                # type_dict = dict(color="white", linewidth=1, zorder=10)
                continue
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                # type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
                continue
            else:
                # type_dict = dict(color="white", linewidth=2, zorder=10)
                continue
        elif ls.attributes["type"] == "pedestrian_marking":
            # type_dict = dict(color="black", linewidth=1, zorder=10, dashes=[5, 10])
            continue
        elif ls.attributes["type"] == "bike_marking":
            # type_dict = dict(color="black", linewidth=1, zorder=10, dashes=[5, 10])
            continue
        elif ls.attributes["type"] == "stop_line":
            # type_dict = dict(color="white", linewidth=3, zorder=10)
            continue
        elif ls.attributes["type"] == "virtual":
            # type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif ls.attributes["type"] == "road_border":
            # type_dict = dict(color="black", linewidth=1, zorder=10)
            # type_dict = dict(color="white", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "guard_rail":
            # type_dict = dict(color="black", linewidth=1, zorder=10)
            continue
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=2)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=2)

        else:
            if ls.attributes["type"] not in unknown_linestring_types:
                unknown_linestring_types.append(ls.attributes["type"])
            continue

        ls_points_x = [pt.x for pt in ls]
        ls_points_y = [pt.y for pt in ls]

        plt.plot(ls_points_x, ls_points_y, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

    lanelets = []
    for ll in laneletmap.laneletLayer:
        points = [[pt.x, pt.y] for pt in ll.polygon2d()]
        polygon = Polygon(points, True)
        lanelets.append(polygon)

    ll_patches = PatchCollection(lanelets, facecolors="lightgray", edgecolors="black", zorder=5)
    axes.add_collection(ll_patches)

    if len(laneletmap.laneletLayer) == 0:
        axes.patch.set_facecolor('lightgrey')

    plt.xticks([]) 
    plt.yticks([])
    plt.axis('off')

    return map_x_bound,map_y_bound


def draw_route_center_line(route_lanelet, current_lanelet, current_state, axes):
    for ll in route_lanelet:
        ls = ll.centerline

        # extend centerline point to meet the min interval distance requirement
        min_dist_require = 1.2 # meter
        extend_ls = geometry.insert_node_to_meet_min_interval(ls, min_dist_require)
     
        # red dashline
        # type_dict = dict(color="red", linewidth=1, zorder=10, linestyle=':')
        ls_points_x = [pt.x for pt in extend_ls]
        ls_points_y = [pt.y for pt in extend_ls]

        current_ego_pt_x = current_state.x
        current_ego_pt_y = current_state.y

        # remove points behind ego vehicle in current lanelet
        if len(route_lanelet)>1 and ll.id == current_lanelet.id:
            for i,j in zip(ls_points_x, ls_points_y):
                ls2end_distance = math.sqrt((ls_points_x[-1] - i)**2 + (ls_points_y[-1] - j)**2)
                ego2end_distance = math.sqrt((current_ego_pt_x - ls_points_x[-1])**2 + (current_ego_pt_y - ls_points_y[-1])**2)

                if not ls2end_distance < ego2end_distance:
                    # waypoint is behind ego vehicle
                    ls_points_x.remove(i)
                    ls_points_y.remove(j)

        else:
            # current lanelet is the last lanelet in route
            for i,j in zip(ls_points_x, ls_points_y):
                ls2end_distance = math.sqrt((ls_points_x[-1] - i)**2 + (ls_points_y[-1] - j)**2)
                ego2end_distance = math.sqrt((current_ego_pt_x - ls_points_x[-1])**2 + (current_ego_pt_y - ls_points_y[-1])**2)

                if not ls2end_distance < ego2end_distance:
                    # waypoint is behind ego vehicle
                    ls_points_x.remove(i)
                    ls_points_y.remove(j)

        # green dashline
        type_dict = dict(color="green", linewidth=3, zorder=10)
        plt.plot(ls_points_x, ls_points_y, **type_dict)


def draw_route(route, axes):
    # render all points in green
    ls_points_x = [pt[0] for pt in route]
    ls_points_y = [pt[1] for pt in route]

    # dashline
    type_dict = dict(color="green", linewidth=3, zorder=10)
    plt.plot(ls_points_x, ls_points_y, **type_dict)


def draw_route_bounds(route_bounds, axes):
    # render all bounds points in green
    ls_points_x = [pt.x for pt in route_bounds]
    ls_points_y = [pt.y for pt in route_bounds]

    # dashline
    type_dict = dict(color="magenta", linewidth=1.5, zorder=10)
    plt.plot(ls_points_x, ls_points_y, **type_dict)

def draw_closet_bound_point(previous_closet_points, current_closet_points, axes):
    if previous_closet_points:
        ls_points_x = [pt[0] for pt in previous_closet_points]
        ls_points_y = [pt[1] for pt in previous_closet_points]

        # circle
        centerline_circle = []
        for i in range(len(ls_points_x)):
            cirl = Circle(xy = (ls_points_x[i], ls_points_y[i]), radius=0.45, alpha=0.5)
            centerline_circle.append(cirl)
        centerline_circle_patches = PatchCollection(centerline_circle, facecolors="magenta", zorder=5)
        axes.add_collection(centerline_circle_patches)
        

    # render current future route in green
    ls_points_x = [pt[0] for pt in current_closet_points]
    ls_points_y = [pt[1] for pt in current_closet_points]

    # green circle
    centerline_circle = []
    for i in range(len(ls_points_x)):
        cirl = Circle(xy = (ls_points_x[i], ls_points_y[i]), radius=0.45, alpha=0.5)
        centerline_circle.append(cirl)
    centerline_circle_patches = PatchCollection(centerline_circle, facecolors="green", zorder=5)
    axes.add_collection(centerline_circle_patches)


def draw_ego_future_route(previous_route_points_dict, current_route_points_dict, axes):
    # re-render previous future route in green
    pr_ls_points_x_dict = dict()
    pr_ls_points_y_dict = dict()

    if previous_route_points_dict:
        for ego_id in previous_route_points_dict.keys():
            pr_ls_points_x_dict[ego_id] = []
            pr_ls_points_y_dict[ego_id] = []
            pr_ls_points_x_dict[ego_id] += [pt[0] for pt in previous_route_points_dict[ego_id]]
            pr_ls_points_y_dict[ego_id] += [pt[1] for pt in previous_route_points_dict[ego_id]]

        # green dashline
        type_dict = dict(color="green", linewidth=3, zorder=10)
        for ego_id in pr_ls_points_x_dict.keys():
            pr_ls_points_x = pr_ls_points_x_dict[ego_id]
            pr_ls_points_y = pr_ls_points_y_dict[ego_id]
            plt.plot(pr_ls_points_x, pr_ls_points_y, **type_dict)
        
    # render current future route in red
    ls_points_x_dict = dict()
    ls_points_y_dict = dict()
    for ego_id in previous_route_points_dict.keys():
        ls_points_x_dict[ego_id] = []
        ls_points_y_dict[ego_id] = []
        ls_points_x_dict[ego_id] += [pt[0] for pt in current_route_points_dict[ego_id]]
        ls_points_y_dict[ego_id] += [pt[1] for pt in current_route_points_dict[ego_id]]

    # red dashline
    type_dict = dict(color="red", linewidth=3, zorder=10)
    for ego_id in ls_points_x_dict.keys():
        ls_points_x = ls_points_x_dict[ego_id]
        ls_points_y = ls_points_y_dict[ego_id]
        plt.plot(ls_points_x, ls_points_y, **type_dict)

def draw_conflict_point(conflict_point_list,axes):
    conflicts = []
    for pt in conflict_point_list:
        # print('draw pt x: ',pt.x,' pt y: ',pt.y)
        cirl = Circle(xy = (pt.x,pt.y), radius=1, alpha=0.5)
        conflicts.append(cirl)
    conflict_patches = PatchCollection(conflicts, facecolors="green", edgecolors="white", zorder=5)
    axes.add_collection(conflict_patches)