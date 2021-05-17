import numpy as np
from queue import PriorityQueue
import numpy.linalg as LA
import networkx as nx

from scipy.spatial import Voronoi
from bresenham import bresenham

def create_grid_and_edges(data, drone_altitude, safety_distance):

    print('[start]create_grid_and_edges')

    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    grid = np.zeros((north_size, east_size))
    points = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            points.append([north - north_min, east - east_min])

    graph = Voronoi(points)

    edges = []
    for v in graph.ridge_vertices:
        r1 = graph.vertices[v[0]]
        r2 = graph.vertices[v[1]]
        cells = list(bresenham(int(r1[0]), int(r1[1]), int(r2[0]), int(r2[1])))
        hit = False

        for c in cells:
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            if grid[c[0], c[1]] == 1:
                hit = True
                break
        if not hit:
            r1 = (r1[0], r1[1])
            r2 = (r2[0], r2[1])
            edges.append((r1, r2))

    print('[end]create_grid_and_edges')
    return grid, edges, int(north_min), int(east_min)


def heuristic(m1, m2):
    return LA.norm(np.array(m2) - np.array(m1))

def closest_point(graph, current_point):

    closest_point = None
    dist = 100000
    for z in graph.nodes:
        d = LA.norm(np.array(z) - np.array(current_point))
        if d < dist:
            closest_point = z
            dist = d
    return closest_point


def find_path(graph, edges, grid_start, grid_goal):

    print('[start]find_path ...')

    G = nx.Graph()
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)

    start_closest = closest_point(G, grid_start)
    print(grid_start, start_closest)
    goal_closest = closest_point(G, grid_goal)
    print(grid_goal, goal_closest)

    p, cost = a_star(G, heuristic, start_closest, goal_closest)

    if len(p) > 0:
        p.insert(0, grid_start)
        p.append(grid_goal)

    _path = []
    for n, e in p:
        _path.append((int(n), int(e)))

    print('[end]find_path ...')

    return _path, cost


def a_star(graph, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']

                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))

    if found:
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost