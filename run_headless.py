import os
import sys

# 强制离屏渲染
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['DISPLAY'] = ':0'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# 设置 PyFleX 路径
os.environ['PYFLEXROOT'] = os.path.join(os.getcwd(), 'PyFleX')
os.environ['PYTHONPATH'] = os.path.join(os.environ['PYFLEXROOT'], 'bindings/build') + ':' + os.environ.get('PYTHONPATH', '')
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.environ['PYFLEXROOT'], 'external/SDL2-2.0.4/lib/x64') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# 执行目标脚本
with open(sys.argv[1]) as f:
    exec(f.read(), {'__name__': '__main__'})
