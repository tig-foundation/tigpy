from tigpy.data import DifficultyParamConfig, FrontierPoint
from tigpy.utils import timeit
from typing import List
import random
import numpy as np

@timeit
def isParetoEfficient(costs):
    """
    Find the pareto-efficient points.
    Adapted from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    :param costs: An (n_points, n_costs) array
    :return: An array of indices of pareto-efficient points.
        Returning mask of (n_points, ) boolean array and a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[np.all(costs == costs[next_point_index], axis=1)] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask, is_efficient

def clipDifficulty(difficulty: List[int], difficulty_parameters: List[DifficultyParamConfig]):
    return [
        min(max(difficulty[i], difficulty_parameters[i].min), difficulty_parameters[i].max)
        for i in range(len(difficulty))
    ]

@timeit
def randomDifficultyOnFrontier(points: List[FrontierPoint], difficulty_parameters: List[DifficultyParamConfig]):
    if len(difficulty_parameters) != 2:
        raise ValueError("Only 2 difficulty parameters are supported")
    if not all(p.frontier_idx == points[0].frontier_idx for p in points):
        raise ValueError("The list of frontier points must have the same frontier_idx")
    min_difficulty = [
        difficulty_parameters[0].min,
        difficulty_parameters[1].min
    ]
    max_difficulty = [
        max(p.difficulty[0] for p in points),
        max(p.difficulty[1] for p in points)
    ]
    
    difficulties = set(tuple(p.difficulty) for p in points)
    # Add points right on the bounds so we can interpolate across the full x and y range
    if not any(d[0] == min_difficulty[0] for d in difficulties):
        difficulties.add((min_difficulty[0], max_difficulty[0] + 1))
        max_difficulty[0] += 1
    if not any(d[1] == min_difficulty[1] for d in difficulties):
        difficulties.add((max_difficulty[1] + 1, min_difficulty[1]))
        max_difficulty[1] += 1
    
    # Interpolate a random x, y point
    if len(difficulties) == 1:
        random_difficulty = list(difficulties)[0]
    else:
        dim = int(0.5 < random.random())
        dim2 = int(not dim)
        difficulties = sorted(difficulties, key=lambda d: d[dim])
        random_difficulty = [None, None]
        random_difficulty[dim] = random.randint(min_difficulty[dim], max_difficulty[dim])
        idx = np.searchsorted(
            [d[dim] for d in difficulties],
            random_difficulty[dim]
        )
        if random_difficulty[dim] == difficulties[idx][dim]: # existing point, no need to interpolate
            random_difficulty[dim2] = difficulties[idx][dim2]
        else:
            random_difficulty[dim2] = int(np.ceil(np.interp(
                random_difficulty[dim], 
                [difficulties[idx - 1][dim], difficulties[idx][dim]], 
                [difficulties[idx - 1][dim2], difficulties[idx][dim2]]
            )))

    return clipDifficulty(random_difficulty, difficulty_parameters)