import time

import torch
from shapely.geometry import Point

from scripts.models.lol.dice_coefficient.Intersection import Intersection
from scripts.models.lol.dice_coefficient.dice_polygon import DicePolygon
from scripts.models.lol.dice_coefficient.geometric import line_intersect, segments_to_vertices


def WeilerAtherton(clip, subject, verbose=False):
    counter = 0
    intersection_map = {}
    clip_segments = clip.lines()
    subj_segments = subject.lines()

    # TODO: it still, although very rarely, gets stuck in an infinite loop.
    # For now, fail safe mechanism that raises exception when taking too long
    # Max time in seconds = max amount of edges in the polygons
    max_time_in_seconds = (0.5) * max(len(clip_segments), len(subj_segments))
    start_time = time.time()

    for clip_line in clip_segments:
        for subj_line in subj_segments:
            intersection_point = line_intersect(clip_line, subj_line)
            if intersection_point is not None:
                counter += 1
                id = str(counter)
                clip_line.add_intersection_id(id)
                subj_line.add_intersection_id(id)
                intersection_map[id] = Intersection(id, intersection_point)

    [s.sort(intersection_map) for s in clip_segments]
    [s.sort(intersection_map) for s in subj_segments]

    for clip_line in clip_segments:
        entering = not subject.as_shapely().contains(clip_line.starting_point())
        for intersection_id in clip_line.intersection_ids:
            intersection_map[intersection_id].entering = entering
            intersection_map[intersection_id].leaving = not entering
            entering = not entering

    # If there are no intersections,
    # one polygon is contained in the other
    if len(intersection_map) == 0:
        clip_point = Point(clip.tensor[0][0].item(), clip.tensor[0][1].item())
        subj_polygon = subject.as_shapely()
        clip_is_within_subj = subj_polygon.contains(clip_point)
        return [DicePolygon(clip)] if clip_is_within_subj else [DicePolygon(subject)]

    clip_vertices = segments_to_vertices(intersection_map, clip_segments)
    subj_vertices = segments_to_vertices(intersection_map, subj_segments)

    visited_intersections = {}
    clips = []

    if verbose:
        print(len(intersection_map), "intersections")

    entrance_to_clip = None
    (current_entrance_index, current_entrance) = (None, None)
    current_clip = []

    while True:

        if time.time() - start_time > max_time_in_seconds:
            raise Exception("Failed to find Weiler-Atherton solution")

        # If no target clipping entrance it set, find first not visited
        if entrance_to_clip is None:
            ### FIND ENTRANCE
            entrances_not_visited = [(i, v) for i, v in enumerate(clip_vertices) if
                                     v.is_intersection and v.entering and v.id not in visited_intersections]
            if len(entrances_not_visited) == 0:
                if verbose:
                    print("No more entrances to visit.")
                break
            (current_entrance_index, current_entrance) = entrances_not_visited[0]
            entrance_to_clip = current_entrance
            if verbose:
                print("Entering clip at", current_entrance.tensor.detach().cpu().numpy())

        ### FIND EXIT IN CLIP
        next_exits_ahead = [(i, v) for i, v in enumerate(clip_vertices) if
                            v.is_intersection and not v.entering and i > current_entrance_index]
        next_exits_before = [(i, v) for i, v in enumerate(clip_vertices) if
                             v.is_intersection and not v.entering and i < current_entrance_index]

        if len(next_exits_ahead) > 0:
            (exit_index, exit_vertex) = next_exits_ahead[0]
            current_clip += clip_vertices[current_entrance_index:exit_index + 1]
        else:
            (exit_index, exit_vertex) = next_exits_before[-1]
            current_clip += clip_vertices[current_entrance_index:] + clip_vertices[:exit_index + 1]

        if verbose:
            print("Exiting clip at", exit_vertex.tensor.detach().cpu().numpy())

        ### SWITCH LIST
        ### START AT THE INDEX OF THE EXIT
        ### FIND NEXT ENTERING EDGE
        entrance_index, start_vertex = \
            [(i, v) for i, v in enumerate(subj_vertices) if v.is_intersection and v.id == exit_vertex.id][0]

        next_entrances_ahead = [(i, v) for i, v in enumerate(subj_vertices) if
                                v.is_intersection and v.entering and i > entrance_index]
        next_entrances_before = [(i, v) for i, v in enumerate(subj_vertices) if
                                 v.is_intersection and v.entering and i < entrance_index]

        if len(next_entrances_ahead) > 0:
            (exit_index, exit_vertex) = next_entrances_ahead[0]
            current_clip += subj_vertices[entrance_index + 1:exit_index]
        else:
            (exit_index, exit_vertex) = next_entrances_before[-1]
            current_clip += subj_vertices[entrance_index + 1:] + subj_vertices[:exit_index]

        if verbose:
            print("Entering subject at", start_vertex.tensor.detach().cpu().numpy())
        if verbose:
            print("Exiting subject subject at", exit_vertex.tensor.detach().cpu().numpy())

        if exit_vertex.same_as(entrance_to_clip):
            if verbose:
                print("[Finished clip]")
                for t in current_clip:
                    print(t.tensor.detach().cpu().numpy())

            clips.append(current_clip)
            for vertex in current_clip:
                if vertex.is_intersection:
                    visited_intersections[vertex.id] = vertex
            current_clip = []
            entrance_to_clip = None
        else:
            (current_entrance_index, current_entrance) = \
            [(i, v) for i, v in enumerate(clip_vertices) if v.is_intersection and v.id == exit_vertex.id][0]
            # CHECK FOR COMPLETION
        if len(visited_intersections) >= len(intersection_map):
            break
    if verbose:
        print(len(clips), "clip(s)", "of lengthts", [len(clip) for clip in clips])

        for i, clip in enumerate(clips):
            print("[Clip", str(i), "]")
            for t in clip:
                print(t.tensor.detach().cpu().numpy())

    return [DicePolygon(torch.stack([t.tensor for t in c])) for c in clips]


if __name__ == "__main__":
    # 30|      ---------------
    #   |      |     top     |
    # 25|      |   __    __  |
    #   |      |  |XX|  |XX| |
    # 15|      ---------------
    # 10|   ______|  |__|  |______
    #   |  |        bottom       |
    # 5 |  |_____________________|
    #   |_____________________________
    #      5     15 20  25 30   40

    top = torch.tensor([
        [10, 15],
        [10, 30],
        [35, 30],
        [35, 15]
    ], requires_grad=True, dtype=torch.float32)

    bottom = torch.tensor([
        [5, 5],
        [5, 10],
        [15, 10],
        [15, 25],
        [20, 25],
        [20, 10],
        [25, 10],
        [25, 25],
        [30, 25],
        [30, 10],
        [40, 10],
        [40, 5],
    ], requires_grad=True, dtype=torch.float32)

    clips = WeilerAtherton(DicePolygon(bottom), DicePolygon(top))

    assert len(clips) == 2
