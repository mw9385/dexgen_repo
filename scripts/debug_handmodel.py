"""Debug script: test HandModel loading step by step."""
import sys
sys.path.insert(0, '/workspace/dexgen/third_party/DexGraspNet/thirdparty/pytorch_kinematics')
sys.path.insert(0, '/workspace/dexgen')

print('1. imports ok')

import pytorch_kinematics as pk
print('2. pk imported')

chain = pk.build_chain_from_mjcf(
    open('/workspace/dexgen/third_party/DexGraspNet/grasp_generation/mjcf/shadow_hand_wrist_free.xml').read()
)
print('3. chain built:', len(chain.get_joint_parameter_names()), 'DOFs')

import torch
chain = chain.to(dtype=torch.float, device='cuda')
print('4. chain on cuda')

from grasp_generation.hand_model import DexGraspNetHandModel
print('5. HandModel class loaded')

model = DexGraspNetHandModel(device='cuda')
print('6. HandModel created OK')
