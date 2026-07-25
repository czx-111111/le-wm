ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/setup_env.sh"

echo "========== 1) NVIDIA / EGL 诊断 =========="
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-'(空)'}"
echo "EGL_GPU=${EGL_GPU:-'(空)'}"
echo "PWD=$PWD"

echo "--- /dev/nvidia* ---"
ls -l /dev/nvidia* 2>/dev/null || echo "(无 nvidia 设备节点)"

echo "--- /dev/dri ---"
ls -l /dev/dri 2>/dev/null || echo "(无 /dev/dri — 必须用 EGL_GPU)"

echo "--- libEGL_nvidia ---"
ldconfig -p 2>/dev/null | grep -i 'libEGL_nvidia' || echo "(未找到)"

echo "--- glvnd egl vendor ---"
ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || true

echo "========== 2) 禁用 Mesa vendor =========="
if [ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
  mv -f /usr/share/glvnd/egl_vendor.d/50_mesa.json \
        /usr/share/glvnd/egl_vendor.d/50_mesa.json.disabled
  echo "已禁用 50_mesa.json"
else
  echo "Mesa vendor 已不在"
fi

echo "========== 3) 用 EGL_GPU 测试 pyflex.init =========="
# 优先 5，再试其它卡
for gpu in 5 0 1 2 3 4 6 7; do
  echo "----- try EGL_GPU=$gpu -----"
  export EGL_GPU=$gpu
  set +e
  out=$(
    python - <<'PY' 2>&1
import os
import numpy as np
import pyflex

print("EGL_GPU=", os.environ.get("EGL_GPU"))
pyflex.set_screenWidth(64)
pyflex.set_screenHeight(64)
pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
pyflex.set_light_fov(70.0)
print("calling pyflex.init(True)...")
pyflex.init(True)
print("SUCCESS: pyflex.init OK with EGL_GPU=%s" % os.environ.get("EGL_GPU"))
PY
  )
  rc=$?
  set -e
  echo "$out"
  if echo "$out" | grep -q "SUCCESS: pyflex.init OK"; then
    echo ""
    echo "=========================================="
    echo "找到可用设置: export EGL_GPU=$gpu"
    echo "请执行:"
    echo "  export EGL_GPU=$gpu"
    echo "  source setup_env.sh"
    echo "  python eval_rope_1.py --smoke-env"
    echo "=========================================="
    exit 0
  fi
  echo "(EGL_GPU=$gpu 失败, exit=$rc)"
  echo ""
done

echo "所有 EGL_GPU 都失败。可重建容器并挂 /dev/dri:"
echo "  docker run --gpus all --device /dev/dri -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ..."
exit 1
