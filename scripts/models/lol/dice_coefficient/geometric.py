import torch


def line_intersect(line_a, line_b):
    """ returns a (x, y) tensor or None if there is no intersection """

    Ax1 = line_a[0][0]
    Ay1 = line_a[0][1]
    Ax2 = line_a[1][0]
    Ay2 = line_a[1][1]
    Bx1 = line_b[0][0]
    By1 = line_b[0][1]
    Bx2 = line_b[1][0]
    By2 = line_b[1][1]

    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return torch.stack([x, y])


def segments_to_vertices(map, segments):
    """
    Transforms a list of segments into a list of the ordered vertices of those segments
    :param map: Map of the intersection
    :param segments: Polygon segments from which vertices are extracted
    :return: Ordered list of vertices, including intersections
    """
    all_vertices = []
    for index, segment in enumerate(segments):
        all_this_segment_vertices = segment.all(map)
        if index == 0:
            all_vertices += all_this_segment_vertices
        elif index == len(segments) - 1:
            all_vertices += all_this_segment_vertices[1:-1]
        else:
            all_vertices += all_this_segment_vertices[1:]
    return all_vertices
