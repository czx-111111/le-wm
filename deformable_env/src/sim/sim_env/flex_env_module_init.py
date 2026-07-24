import os
import numpy as np
import pyflex
import gym
import math
from scipy.spatial.distance import cdist

import pybullet as p
import pybullet_data

from .robot_env import FlexRobotHelper

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

from .flex_scene import FlexScene
from .cameras import Camera
from ..utils import fps_with_idx, quatFromAxisAngle, find_min_distance, rand_float

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../../../"))

# ============================================================
# 🔧 关键修复：在模块级别初始化 pyflex
# 这样在类实例化之前，pyflex 就已经初始化好了
# ============================================================
print("[MODULE] 在模块级别初始化 pyflex...")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 设置 pyflex 参数
pyflex.set_screenWidth(224)
pyflex.set_screenHeight(224)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.0)

# 初始化 pyflex
try:
    pyflex.init(True)
    print("[MODULE] pyflex.init 成功！")
except Exception as e:
    print(f"[MODULE] pyflex.init 失败: {e}")
    raise

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        print("  [DEBUG] FlexEnv.__init__ 开始")
        super().__init__()
        print("  [DEBUG] super().__init__() 完成")

        self.dataset_config = config["dataset"]
        print(f"  [DEBUG] dataset_config keys: {list(self.dataset_config.keys())}")

        print("  [DEBUG] 设置 env component...")
        self.obj = self.dataset_config["obj"]
        self.obj_params = self.dataset_config["obj_params"]
        self.scene = FlexScene()
        print(f"  [DEBUG] obj: {self.obj}")

        print("  [DEBUG] 设置 robot arm...")
        self.flex_robot_helper = FlexRobotHelper()
        self.end_idx = self.dataset_config["robot_end_idx"]
        self.num_dofs = self.dataset_config["robot_num_dofs"]
        self.robot_speed_inv = self.dataset_config["robot_speed_inv"]
        print(f"  [DEBUG] end_idx: {self.end_idx}, num_dofs: {self.num_dofs}")

        # pyflex 已经在模块级别初始化，这里不再重复初始化
        print("  [DEBUG] pyflex 已在模块级别初始化，跳过...")

        # 初始化 pybullet
        print("  [DEBUG] 初始化 pybullet...")
        # 先断开任何可能存在的连接
        try:
            p.disconnect()
        except:
            pass
        physicsClient = p.connect(p.DIRECT)
        print(f"  [DEBUG] physicsClient: {physicsClient}")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        print(f"  [DEBUG] planeId: {self.planeId}")

        print("  [DEBUG] 设置 camera...")
        self.screenWidth = self.dataset_config["screenWidth"]
        self.screenHeight = self.dataset_config["screenHeight"]
        self.camera = Camera(self.screenWidth, self.screenHeight)
        self.camera_view = self.dataset_config["camera_view"]
        print(f"  [DEBUG] camera_view: {self.camera_view}")

        print("  [DEBUG] 定义 action space...")
        self.action_dim = self.dataset_config["action_dim"]
        self.action_space = self.dataset_config["action_space"]

        print("  [DEBUG] 初始化统计变量...")
        self.count = 0
        self.imgs_list = []
        self.particle_pos_list = []
        self.eef_states_list = []

        self.fps = self.dataset_config["fps"]
        self.fps_number = self.dataset_config["fps_number"]

        self.gripper = self.dataset_config["gripper"]
        self.stick_len = self.dataset_config["pusher_len"]
        print(f"  [DEBUG] gripper: {self.gripper}, stick_len: {self.stick_len}")

        print("  [DEBUG] 调用 scene.set_scene...")
        self.scene.set_scene(self.obj, self.obj_params)
        print("  [DEBUG] scene.set_scene 完成")

        print("  [DEBUG] 调用 camera.set_init_camera...")
        self.camera.set_init_camera(self.camera_view)
        print("  [DEBUG] camera.set_init_camera 完成")

        print("  [DEBUG] 调用 add_table...")
        self.add_table()
        print("  [DEBUG] add_table 完成")

        print("  [DEBUG] 调用 add_robot...")
        self.add_robot()
        print("  [DEBUG] add_robot 完成")
        print("  [DEBUG] FlexEnv.__init__ 全部完成")
    
    def add_table(self):
        """添加桌子"""
        # 这里需要实现实际的 add_table 逻辑
        pass
    
    def add_robot(self):
        """添加机器人"""
        # 这里需要实现实际的 add_robot 逻辑
        pass
    
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]
        shape_states = np.zeros((n_table + n_robot_links, 14))
        shape_states[:n_table] = self.table_shape_states
        return shape_states
