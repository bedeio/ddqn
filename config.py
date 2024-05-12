from agent import Config
from torch.functional import F

def select_config(env_name, configs, default=None):
    for key in configs:
        if env_name.startswith(key):
            return configs[key]
    
    print(f"No configuration found for environment: {env_name}")
    default

agent_configs = {
    "WindyCartPole": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-4, UPDATE_EVERY=4, LOSS=F.mse_loss),
    "CartPole": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-4, UPDATE_EVERY=4, LOSS=F.mse_loss),
    "LunarLander": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.99, TAU=1e-2, LR=1e-3, UPDATE_EVERY=4, LOSS=F.smooth_l1_loss),
    "Taxi": Config(DDQN=True, BUFFER_SIZE=int(7e5), BATCH_SIZE=512, GAMMA=0.98, TAU=1e-2, LR=4e-3, UPDATE_EVERY=4, LOSS=F.mse_loss)
}

solved_scores = {
    # https://github.com/openai/gym/wiki/Leaderboard
    # https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards
    "WindyCartPole": 500,
    "CartPole": 500,
    "LunarLander": 250, 
    "Taxi": 8.5
}

robust_kwargs = {
    "WindyCartPole": [
        {
            "turbulence_power": 0.1,
            "wind_power": 0.0
        },
        {
            "turbulence_power": 0.1,
            "wind_power": 0.2
        },
        {
            "turbulence_power": 0.1,
            "wind_power": 0.4
        },
        {
            "turbulence_power": 0.1,
            "wind_power": 0.6
        },
        {
            "turbulence_power": 0.1,
            "wind_power": 0.8
        },
        {
            "turbulence_power": 0.1,
            "wind_power": 1.0
        },
    ],
    "LunarLander": [
        {
            "turbulence_power": 2.0,
            "enable_wind": True,
            "wind_power": 0.0,
            "continuous": False,
            "gravity": -10.0,
        },
        {
            "turbulence_power": 2.0,
            "enable_wind": True,
            "wind_power": 5.0,
            "continuous": False,
            "gravity": -10.0,
        },
        {
            "turbulence_power": 2.0,
            "enable_wind": True,
            "wind_power": 10.0,
            "continuous": False,
            "gravity": -10.0,
        },
        {
            "turbulence_power": 2.0,
            "enable_wind": True,
            "wind_power": 15.0,
            "continuous": False,
            "gravity": -10.0,
        },
        {
            "turbulence_power": 2.0,
            "enable_wind": True,
            "wind_power": 20.0,
            "continuous": False,
            "gravity": -10.0,
        }
    ]
}

