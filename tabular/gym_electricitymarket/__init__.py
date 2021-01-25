from gym.envs.registration import register

register(
    id='ElectricityMarket-v0',
    entry_point='gym_electricitymarket.envs:ElectricityMarket',
)