from gym_dogfight.envs.registration import registry, register, make, spec

# register(
#     id="oneonone-v0",
#     entry_point="gym.envs.dogfightEnv:oneVSoneEnv",
# )

register(
    id="human_expert_env-v0",
    entry_point="gym_dogfight.envs.dogfightEnv:human_expert_env",
)
register(
    id="data_collection-v0",
    entry_point="gym_dogfight.envs.dogfightEnv:data_collection",
)
#
register(
    id="onevsone_ap-v0",
    entry_point="gym_dogfight.envs.dogfightEnv:oneVSoneEnv_ap",
)
