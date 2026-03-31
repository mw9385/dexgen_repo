"""Debug script: test HandModel loading step by step."""
import os
import sys
import subprocess

sys.path.insert(0, '/workspace/dexgen/third_party/DexGraspNet/thirdparty/pytorch_kinematics')
sys.path.insert(0, '/workspace/dexgen')

# Auto-install missing dependencies
for pkg in ['transforms3d', 'transformations', 'lxml']:
    try:
        __import__(pkg)
    except ImportError:
        print(f'Installing missing package: {pkg}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

print('1. imports ok')

import pytorch_kinematics as pk
print('2. pk imported')

# MJCF parser resolves mesh paths relative to CWD
mjcf_dir = '/workspace/dexgen/third_party/DexGraspNet/grasp_generation'
mjcf_file = os.path.join(mjcf_dir, 'mjcf', 'shadow_hand_wrist_free.xml')

original_cwd = os.getcwd()
os.chdir(mjcf_dir)
chain = pk.build_chain_from_mjcf(open(mjcf_file).read())
os.chdir(original_cwd)
print('3. chain built:', len(chain.get_joint_parameter_names()), 'DOFs')

import torch
chain = chain.to(dtype=torch.float, device='cuda')
print('4. chain on cuda')

from grasp_generation.hand_model import DexGraspNetHandModel
print('5. HandModel class loaded')

model = DexGraspNetHandModel(device='cuda')
print('6. HandModel created OK')
