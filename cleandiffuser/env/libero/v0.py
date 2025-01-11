from cleandiffuser.env.libero.libero_env import LiberoEnv


class LiberoObjectEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_object"


class LiberoGoalEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_goal"


class LiberoSpatialEnv(LiberoEnv):
    TASK_SUITE_NAME = "libero_spatial"


class Libero10Env(LiberoEnv):
    TASK_SUITE_NAME = "libero_10"


class Libero90Env(LiberoEnv):
    TASK_SUITE_NAME = "libero_90"
