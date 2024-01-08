from tigpy.data import DifficultyParameterConfig, FrontierPoint
from tigpy.utils import timeit
from typing import List
import random
import math

@timeit
def isParetoEfficient(costs):
    import numpy as np
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

def clipDifficulty(difficulty: List[int], difficulty_parameters: List[DifficultyParameterConfig]):
    return [
        min(max(difficulty[i], difficulty_parameters[i].min_value), difficulty_parameters[i].max_value)
        for i in range(len(difficulty))
    ]

@timeit
def randomDifficultyOnFrontier(frontier_points: List[FrontierPoint], difficulty_parameters: List[DifficultyParameterConfig], frontier_idx: int = 0) -> List[int]:
    import numpy as np
    if len(difficulty_parameters) != 2:
        raise ValueError("Only 2 difficulty parameters are supported")
    frontier_points = [p for p in frontier_points if p.frontier_idx == frontier_idx]
    min_difficulty = [
        difficulty_parameters[0].min_value,
        difficulty_parameters[1].min_value
    ]
    max_difficulty = [
        max(p.difficulty[0] for p in frontier_points),
        max(p.difficulty[1] for p in frontier_points)
    ]
    
    difficulties = set(tuple(p.difficulty) for p in frontier_points)
    # Add points right on the bounds so we can interpolate across the full x and y range
    if not any(d[0] == min_difficulty[0] for d in difficulties):
        difficulties.add((min_difficulty[0], max_difficulty[1] + 1))
        max_difficulty[1] += 1
    if not any(d[1] == min_difficulty[1] for d in difficulties):
        difficulties.add((max_difficulty[0] + 1, min_difficulty[1]))
        max_difficulty[0] += 1
    
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
            random_difficulty[dim2] = int(np.round(np.interp(
                random_difficulty[dim], 
                [difficulties[idx - 1][dim], difficulties[idx][dim]], 
                [difficulties[idx - 1][dim2], difficulties[idx][dim2]]
            )))

    random_difficulty = clipDifficulty(random_difficulty, difficulty_parameters)
    snap_point = set(
        tuple(d)
        for dim in range(len(difficulty_parameters))
        for d in difficulties    
        if random_difficulty[dim] == d[dim]
    )
    if len(snap_point):
        random_difficulty = list(random.choice(list(snap_point)))
    return random_difficulty


@timeit
def calcUpperLowerFrontier(
    points: List[FrontierPoint], 
    difficulty_parameters: List[DifficultyParameterConfig], 
    num_qualifiers_threshold: int, 
    max_difficulty_multiplier: float,
    frontier_idx: int = 0,
):
    num_qualifiers = sum(p.num_qualifiers for p in points if p.frontier_idx is not None)
    difficulty_multiplier = min(num_qualifiers / num_qualifiers_threshold, max_difficulty_multiplier)
    frontier1 = set(tuple(p.difficulty) for p in points if p.frontier_idx == frontier_idx)
    min_difficulty = tuple(
        param.min_value
        for param in difficulty_parameters
    )
    if len(frontier1) == 0:
        lower = {min_difficulty}
        upper = {min_difficulty}
    else:
        # add edge points for frontier1
        for i, param in enumerate(difficulty_parameters):
            v = max(d[i] for d in frontier1)
            d = min_difficulty[:i] + (v,) + min_difficulty[i + 1:]
            if d not in frontier1:
                frontier1.add(
                    min_difficulty[:i] + (min(v + 1, param.max_value),) + min_difficulty[i + 1:]
                )
        # do difficulty multiplier for all other points
        offsets = [
            [
                (v - param.min_value + 1) * difficulty_multiplier
                for v, param in zip(d, difficulty_parameters)
            ]
            for d in frontier1
        ]
        frontier2 = set(
            tuple(
                max(min(param.min_value - 1 + math.ceil(o), param.max_value), param.min_value)
                for param, o in zip(difficulty_parameters, offset)
            )
            for offset in offsets
        )
        # add edge points for frontier2
        for i, param in enumerate(difficulty_parameters):
            v = max(d[i] for d in frontier2)
            d = min_difficulty[:i] + (v,) + min_difficulty[i + 1:]
            if d not in frontier2:
                frontier2.add(
                    min_difficulty[:i] + (min(v + 1, param.max_value),) + min_difficulty[i + 1:]
                )
        if difficulty_multiplier >= 1:
            upper, lower = frontier2, frontier1
        else:
            upper, lower = frontier1, frontier2
    
    upper_filtered = set()
    for i in range(len(difficulty_parameters)):
        unique = {}
        for p in upper:
            k = p[:i] + (None,) + p[i + 1:]
            unique.setdefault(k, p[i])
            unique[k] = max(unique[k], p[i])
        for p, v in unique.items():
            upper_filtered.add(p[:i] + (v,) + p[i + 1:])
            
    lower_filtered = set()
    for i in range(len(difficulty_parameters)):
        unique = {}
        for p in lower:
            k = p[:i] + (None,) + p[i + 1:]
            unique.setdefault(k, p[i])
            unique[k] = min(unique[k], p[i])
        for p, v in unique.items():
            lower_filtered.add(p[:i] + (v,) + p[i + 1:])

    return (
        [list(p) for p in sorted(upper_filtered)], 
        [list(p) for p in sorted(lower_filtered)]
    )