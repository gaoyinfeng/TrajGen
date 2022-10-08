
from utils import tracks_vis

def update_ego_others_param(timestamp, ego_info, other_info, ghost_info, pedestrian_dictionary):
    # update ego vehicles
    tracks_vis.update_objects_ego(ego_info['shape'], ego_info['state'], ego_info['polygon'])

    # update ghost vehicle
    ghost_track_dict = ego_info['track']
    tracks_vis.update_objects_ghost(timestamp, ghost_info['polygon'], ghost_info['state'], ghost_track_dict)

    # update others without postion conflict with ego vehicle
    tracks_vis.update_objects_without_ego_and_conflict(timestamp, ego_info['polygon'], other_info['polygon'], other_info['state'], other_info['track'], pedestrian_dictionary)
    
    
def render_vehicles_with_highlight(ego_info, other_info, ghost_info, highlight_vehicle_id_list, text_dict, axes, fig, ghost_vis):
    # render ego vehicle
    tracks_vis.render_objects_ego(ego_info['patch'], ego_info['polygon'], ego_info['state'], text_dict, axes)

    # render others vehicle with highlight surrounding
    tracks_vis.render_objects_without_ego_and_conflict_with_highlight(other_info['patch'], other_info['polygon'], other_info['state'], highlight_vehicle_id_list, text_dict, axes)
    
    if ghost_vis:
        # render ego ghost vehicle
        tracks_vis.render_objects_ghost(ghost_info['patch'], ghost_info['polygon'], ghost_info['state'], text_dict, axes, render_as_ego=False)

    fig.canvas.draw()
