import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYFLEX_HEADLESS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))

from deformable_env.FlexEnvWrapper import FlexEnvWrapper

print("尝试创建绳子环境...")
env = FlexEnvWrapper(object_name='rope')
print("✅ 环境创建成功")
# env.close()
