import random
import cv2
import numpy as np
from scipy.spatial import KDTree

def plotPaths(paths, path_scale=1, upscale = 10):
    img = np.ones((297 * upscale, 420 * upscale, 3), np.uint8) * 255
    img = np.zeros((297 * upscale, 420 * upscale, 3), np.uint8) 

    for path in paths:

        color = (128 * int(random.random() + 128), 128 * int(random.random() + 128), 128 * int(random.random() + 128))
        # color = (255,255,255)
        for i in range(len(path) - 1):
            a = (
                round(path[i][0] * upscale * path_scale),
                round(path[i][1] * upscale * path_scale)
            )
            
            b = (
                round(path[i + 1][0] * upscale * path_scale),
                round(path[i + 1][1] * upscale * path_scale)
            )

            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        
        for i in range(len(path)):
            a = (
                round(path[i][0] * upscale * path_scale),
                round(path[i][1] * upscale * path_scale),
            )
            if a[1] < img.shape[0] and a[0] < img.shape[1]:
                img[a[1], a[0]] = [0,255,0]
                
        if False:
            for p in path:
                a = (round(p[0] * upscale * path_scale), round(p[1] * upscale * path_scale))
                if a[0] < img.shape[1] and a[1] < img.shape[0]:
                    img[a[1], a[0], 0] = 255

    return img

def add_path_with_merge(current_paths, new_path, add=True, threshold = 5) :
    if len(current_paths) == 0:
        return [new_path]

    points = []
    for path in current_paths:
        points.append(path[0]) # beginning
        points.append(path[-1]) # end

    tree = KDTree(points)

    end_point = new_path[-1]

    distance, idx = tree.query(end_point, k=2)
    other_idx = idx[1] // 2
    other_is_front = idx[1] %2 == 0

    if distance[1] < threshold :

        other_path = current_paths[other_idx]
        if other_is_front: 
            new_path = new_path + other_path
            current_paths.pop(other_idx)
            current_paths.append(new_path)
            return current_paths
        else:
            new_path = new_path + other_path[::-1]
            current_paths.pop(other_idx)
            current_paths.append(new_path)
            return current_paths

    start_point = new_path[0]
        
    distance, idx = tree.query(start_point, k=2)
    other_idx = idx[1] // 2
    other_is_front = idx[1] %2 == 0
    if distance[1] < threshold :

        other_path = current_paths[other_idx]
        if other_is_front: 
            new_path = other_path[::-1] + new_path
            current_paths.pop(other_idx)
            current_paths.append(new_path)
            return current_paths
        else:
            new_path = other_path + new_path
            current_paths.pop(other_idx)
            current_paths.append(new_path)
            return current_paths

    if add:
        current_paths.append(new_path)

    return current_paths

def merge_one_path_kd(paths, threshold = 5):
    points = []
    for i in range(len(paths)):
        points.append(paths[i][0])
        points.append(paths[i][-1])

    tree = KDTree(points)

    for i in range(len(paths)):
        end_point = paths[i][-1]

        distance, idx = tree.query(end_point, k=2)
        other_idx = idx[1] // 2
        other_is_front = idx[1] %2 == 0
        if distance[1] < threshold and other_idx != i:

            other_path = paths[other_idx]
            if other_is_front: 
                paths[i] = paths[i] + other_path
                paths.pop(other_idx)
                break
            else:
                paths[i] = paths[i] + other_path[::-1]
                paths.pop(other_idx)
                break

        start_point = paths[i][0]
            
        distance, idx = tree.query(start_point, k=2)
        other_idx = idx[1] // 2
        other_is_front = idx[1] %2 == 0
        if distance[1] < threshold and other_idx != i:

            other_path = paths[other_idx]
            if other_is_front: 
                paths[i] = other_path[::-1] + paths[i]
                paths.pop(other_idx)
                break
            else:
                paths[i] = other_path + paths[i]
                paths.pop(other_idx)
                break

    return paths

def merge_one_path(paths, threshold = 5):

    for i in range(len(paths)):
        for j in range(len(paths)):
            if i == j: continue

            start1, end1 = paths[i][0], paths[i][-1]
            start2, end2 = paths[j][0], paths[j][-1]

            if (np.linalg.norm(np.array(end1) - np.array(start2)) < threshold ):
                new_paths = [paths[k] for k in range(len(paths)) if k!=i and k!=j]
                new_paths.append(paths[i]+paths[j])
                return new_paths

            if (np.linalg.norm(np.array(start1) - np.array(start2)) < threshold ):
                new_paths = [paths[k] for k in range(len(paths)) if k!=i and k!=j]
                new_paths.append(paths[i][::-1]+paths[j])
                return new_paths

            if (np.linalg.norm(np.array(end1) - np.array(end2)) < threshold ):
                new_paths = [paths[k] for k in range(len(paths)) if k!=i and k!=j]
                new_paths.append(paths[i]+paths[j][::-1])
                return new_paths

            if (np.linalg.norm(np.array(start1) - np.array(end2)) < threshold ):
                new_paths = [paths[k] for k in range(len(paths)) if k!=i and k!=j]
                new_paths.append(paths[j] + paths[i])
                return new_paths
            

    return paths

def close_loops(paths, threshold = 10):

    for i in range(len(paths)):
            start1, end1 = paths[i][0], paths[i][-1]
            if (np.linalg.norm(np.array(start1) - np.array(end1)) < threshold ):
                paths[i].append(start1)
    return paths

def smooth_closed_path(points, window_size=10):
    points_array = np.array(points)
    num_points = len(points)
    smoothed_points = []

    for i in range(num_points):
        # Calculate the indices for the moving window, considering wrap-around
        indices = [(i + j - window_size // 2) % num_points for j in range(window_size)]
        window_points = points_array[indices]
        avg_point = np.mean(window_points, axis=0)
        smoothed_points.append(avg_point)

    return smoothed_points

def smooth_path(points, window_size=2):
    points_array = np.array(points)
    smoothed_points = []

    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        window_points = points_array[start:end]
        avg_point = np.mean(window_points, axis=0)
        smoothed_points.append(avg_point)

    return smoothed_points

def smooth_all_2(paths, amount = 0.5) :
    smoothed_paths = []

    for k in range(len(paths)):
        smoothed = [paths[k][0]]
        for i in range(1, len(paths[k])-1):
            target = (paths[k][i-1]+paths[k][i+1])/2
            smoothed.append(paths[k][i] * (1 - amount) + target * amount)
        smoothed.append(paths[k][-1])

        path_is_closed =  np.array_equal(paths[k][0], paths[k][-1])
        
        if (path_is_closed):
            target = (paths[k][1]+paths[k][-2])/2
            smoothed[0] = smoothed[0] * (1 - amount) + target * amount
            smoothed[-1] = smoothed[0] 
        smoothed_paths.append(np.array(smoothed))
    
    return smoothed_paths

def smooth_all(paths, amount = 0.5) :
    result = []
    for path in paths:
        path = np.array(path)
 
        if np.all(path[0] == path[-1]) : 
            result.append(smooth_closed_path(path, amount = 0.5))
        else:
            result.append(smooth_path(path,amount = 0.5))
    return result

def simplify_all(paths, threshold=0):
    for i in range(len(paths)):
        paths[i] = rdp(paths[i], threshold)
    return (paths)

def rdp(points, epsilon):
    """
    Simplifies a path using the Ramer-Douglas-Peucker algorithm.

    :param points: List of 2D points.
    :param epsilon: Distance threshold for simplification.
    :return: Simplified list of 2D points.
    """
    points = np.array(points)

    # Find the point with the maximum distance from the line formed by the first and last points
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1, end - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    # If the maximum distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        # Recursive call
        results1 = rdp(points[:index+1], epsilon)
        results2 = rdp(points[index:], epsilon)

        # Build the result list
        result = np.vstack((results1[:-1], results2))
    else:
        result = np.array([points[0], points[-1]])

    return result

def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line.

    :param point: The point (x, y).
    :param line_start: The start of the line (x, y).
    :param line_end: The end of the line (x, y).
    :return: The perpendicular distance.
    """
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    else:
        return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

