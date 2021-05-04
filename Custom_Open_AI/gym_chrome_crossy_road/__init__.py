from gym.envs.registration import register

register(
    id='ChromeCrossyRoad-v0', 
    entry_point='gym_chrome_crossy_road.envs:ChromeCrossyRoadEnv'
)