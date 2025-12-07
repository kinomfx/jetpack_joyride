import math
from typing import Dict, Optional, Tuple

import environment as env
try:
    import numpy as np
except Exception:
    np = None


def find_nearest_obstacle(agent_x: float, obstacles) -> Optional[dict]:
    """Return nearest obstacle details or None.

    Details: rotation, rotation_speed, center_x, center_y, top, bottom
    """
    if not obstacles:
        return None
    nearest = None
    min_dist = float("inf")
    for obs in obstacles:
        # use obstacle center x distance
        dist = abs(obs.x - agent_x)
        if dist < min_dist:
            min_dist = dist
            top = obs.y - obs.height / 2
            bottom = obs.y + obs.height / 2
            nearest = {
                "rotation": getattr(obs, "rotation", 0),
                "rotation_speed": getattr(obs, "rotation_speed", 0),
                "center_x": obs.x,
                "center_y": obs.y,
                "top": top,
                "bottom": bottom,
            }
    return nearest


def find_nearest_coin(agent_x: float, coins) -> Optional[dict]:
    """Return nearest coin details or None. Details: x,y"""
    if not coins:
        return None
    nearest = None
    min_dist = float("inf")
    for c in coins:
        dist = abs(c.x - agent_x)
        if dist < min_dist:
            min_dist = dist
            nearest = {"x": c.x, "y": c.y}
    return nearest


def extract_state(game) -> Dict[str, Optional[Tuple[float, ...]]]:
    """Extract a compact state dict from the running `Game` instance.

    Returns keys:
      - speed_multiplier: float
      - agent_pos: (x, y)
      - nearest_obstacle: {rotation, rotation_speed, center_x, center_y, top, bottom} or None
      - nearest_coin: {x, y} or None
    """
    state = {}
    # speed multiplier (if Spawner has it)
    spawner = getattr(game, "spawner", None)
    speed = None
    if spawner is not None:
        speed = getattr(spawner, "speed_multiplier", 1.0)
    else:
        # fallback based on score if available
        score = getattr(game, "score", 0)
        speed = 1.0 + (score / env.SPEED_SCORE_DIVISOR)
        speed = min(speed, env.MAX_SPEED_MULTIPLIER)

    state["speed_multiplier"] = float(speed)

    # agent position
    agent = getattr(game, "agent", None)
    if agent is not None:
        state["agent_pos"] = (float(agent.circle_pos[0]), float(agent.circle_pos[1]))
    else:
        state["agent_pos"] = None

    # nearest obstacle and coin — query spawner if present
    nearest_obs = None
    nearest_coin = None
    if spawner is not None:
        nearest_obs = find_nearest_obstacle(state["agent_pos"][0], spawner.get_obstacles())
        nearest_coin = find_nearest_coin(state["agent_pos"][0], spawner.get_coins())

    state["nearest_obstacle"] = nearest_obs
    state["nearest_coin"] = nearest_coin

    return state


__all__ = ["extract_state", "find_nearest_obstacle", "find_nearest_coin"]


def extract_state_vector(game, k_obs=2, k_coins=3, max_rot_speed=10.0, max_ttc=300.0):
    """Return a normalized flat state vector for RL.

    Layout:
      [speed_norm, agent_y_norm, agent_vy_norm,
       obs1: (exist, dx, dy, w, h, rot, rot_spd, ttc), ..., obs_k
       coin1: (exist, dx, dy, dist), ..., coin_k]

    Missing numeric values use sentinel -1.0; exist flag is 0.0/1.0.
    Returns numpy array of dtype float32.
    """
    if np is None:
        raise RuntimeError("numpy is required for extract_state_vector")

    sentinel = -1.0
    agent = getattr(game, "agent", None)
    spawner = getattr(game, "spawner", None)
    speed = 1.0
    if spawner is not None:
        speed = getattr(spawner, "speed_multiplier", 1.0)
    else:
        score = getattr(game, "score", 0)
        speed = 1.0 + (score / env.SPEED_SCORE_DIVISOR)
        speed = min(speed, env.MAX_SPEED_MULTIPLIER)

    # normalize speed to [0..1]
    if env.MAX_SPEED_MULTIPLIER > 1.0:
        speed_norm = (speed - 1.0) / (env.MAX_SPEED_MULTIPLIER - 1.0)
    else:
        speed_norm = 0.0

    # agent features
    if agent is None:
        agent_x = env.SCREEN_WIDTH / 2
        agent_y = env.SCREEN_HEIGHT / 2
        agent_vy = 0.0
    else:
        agent_x = float(agent.circle_pos[0])
        agent_y = float(agent.circle_pos[1])
        agent_vy = float(getattr(game, "_agent_vy", 0.0))

    agent_y_norm = agent_y / env.SCREEN_HEIGHT
    # normalize agent_vy by an assumed max (use UPWARD_FORCE*3)
    vy_div = max(1.0, env.UPWARD_FORCE * 3.0)
    agent_vy_norm = max(-1.0, min(1.0, agent_vy / vy_div))

    # prepare arrays
    obs_fields = 8
    coin_fields = 4
    vec_len = 3 + k_obs * obs_fields + k_coins * coin_fields
    vec = np.full((vec_len,), sentinel, dtype=np.float32)
    idx = 0
    vec[idx] = float(speed_norm)
    idx += 1
    vec[idx] = float(agent_y_norm)
    idx += 1
    vec[idx] = float(agent_vy_norm)
    idx += 1

    # collect obstacles
    obstacles = []
    if spawner is not None:
        obstacles = list(spawner.get_obstacles())

    # prefer obstacles ahead of agent, then nearest by absolute distance
    ahead = [o for o in obstacles if (o.x - agent_x) >= 0]
    ahead.sort(key=lambda o: (o.x - agent_x))
    behind = [o for o in obstacles if (o.x - agent_x) < 0]
    behind.sort(key=lambda o: abs(o.x - agent_x))
    ordered = ahead + behind

    for i in range(k_obs):
        if i < len(ordered):
            o = ordered[i]
            exist = 1.0
            dx = float(o.x - agent_x) / env.SCREEN_WIDTH
            dx = max(-1.0, min(1.0, dx))
            dy = float(o.y - agent_y) / env.SCREEN_HEIGHT
            dy = max(-1.0, min(1.0, dy))
            w = float(getattr(o, "width", env.OBSTACLE_WIDTH)) / env.SCREEN_WIDTH
            h = float(getattr(o, "height", env.OBSTACLE_HEIGHT)) / env.SCREEN_HEIGHT
            # rotation normalized to [-1,1]
            rot = float(getattr(o, "rotation", 0.0))
            rot = ((rot + 180.0) % 360.0) - 180.0
            rot_norm = max(-1.0, min(1.0, rot / 180.0))
            rot_spd = float(getattr(o, "rotation_speed", 0.0)) / max_rot_speed
            rot_spd = max(-1.0, min(1.0, rot_spd))
            # time to collision (frames) if ahead
            if (o.x - agent_x) > 0 and spawner is not None:
                rel_speed = env.OBSTACLE_SPEED * max(0.01, getattr(spawner, "speed_multiplier", 1.0))
                ttc = (o.x - agent_x) / rel_speed
                ttc_norm = max(0.0, min(1.0, ttc / max_ttc))
            else:
                ttc_norm = -1.0

            slot = [exist, dx, dy, w, h, rot_norm, rot_spd, ttc_norm]
        else:
            slot = [0.0] + [sentinel] * (obs_fields - 1)

        for v in slot:
            vec[idx] = float(v)
            idx += 1

    # collect coins
    coins = []
    if spawner is not None:
        coins = list(spawner.get_coins())
    # sort coins by horizontal distance ahead first
    ahead_c = [c for c in coins if (c.x - agent_x) >= 0]
    ahead_c.sort(key=lambda c: (c.x - agent_x))
    behind_c = [c for c in coins if (c.x - agent_x) < 0]
    behind_c.sort(key=lambda c: abs(c.x - agent_x))
    ordered_coins = ahead_c + behind_c

    screen_diag = math.hypot(env.SCREEN_WIDTH, env.SCREEN_HEIGHT)
    for j in range(k_coins):
        if j < len(ordered_coins):
            c = ordered_coins[j]
            exist = 1.0
            dx = float(c.x - agent_x) / env.SCREEN_WIDTH
            dx = max(-1.0, min(1.0, dx))
            dy = float(c.y - agent_y) / env.SCREEN_HEIGHT
            dy = max(-1.0, min(1.0, dy))
            dist = math.hypot(c.x - agent_x, c.y - agent_y) / screen_diag
            dist = max(0.0, min(1.0, dist))
            slot = [exist, dx, dy, dist]
        else:
            slot = [0.0] + [sentinel] * (coin_fields - 1)
        for v in slot:
            vec[idx] = float(v)
            idx += 1

    return vec
