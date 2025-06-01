from gymnasium.envs.registration import register

def _register_cellular():
    register(
        id='HelicopterCellularAutomaton-v0',
        entry_point='gym_cellular.environment.helicopter_env:HelicopterEnv',
    )