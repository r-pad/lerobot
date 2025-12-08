from diffusion_policy_3d.model.vision.pointnet_extractor import PointNetEncoderXYZ
import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

with open("data/test_dp3_encoder.pkl", "rb") as f:
    all_pc, all_handle_positions, all_eef_positions, all_num_handle_point_in_gripper_list = pickle.load(f)

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(0)

all_pc = np.array(all_pc)
all_handle_positions = np.array(all_handle_positions)
all_eef_positions = np.array(all_eef_positions)
all_num_handle_point_in_gripper_list = np.array(all_num_handle_point_in_gripper_list)
distance_eef_to_handle = np.linalg.norm(all_eef_positions - all_handle_positions, axis=1)

encoder = PointNetEncoderXYZ(
        in_channels=3,
        out_channels=3,
        use_layernorm=True,
        final_norm='layernorm',
        use_projection=True,
    )
encoder = encoder.cuda()
optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

train_ratio = 0.9
batch_size = 128
train_pc = all_pc[:int(train_ratio * len(all_pc))]
val_pc = all_pc[int(train_ratio * len(all_pc)):]

target = all_handle_positions
train_target = target[:int(train_ratio * len(target))]
val_target = target[int(train_ratio * len(target)):]

num_train_batch = len(train_pc) // batch_size
num_val_batch = len(val_pc) // batch_size

num_train_epochs = 300
train_losses = []
test_losses = []
pbar = tqdm(range(num_train_epochs))
for ep in pbar:
    pbar.set_description("Epoch {}".format(ep))
    train_loss = 0
    for idx in range(num_train_batch):
        start_idx = idx * batch_size
        end_idx = (idx + 1) * batch_size
        pc_batch = train_pc[start_idx:end_idx]
        targ = train_target[start_idx:end_idx]
        pc_batch = torch.tensor(pc_batch).float().cuda()
        targ = torch.tensor(targ).float().cuda()
        pred = encoder(pc_batch)
        loss = torch.mean((pred - targ) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= num_train_batch
    
    
    val_loss = 0
    for idx in range(num_val_batch):
        start_idx = idx * batch_size
        end_idx = (idx + 1) * batch_size
        pc_batch = val_pc[start_idx:end_idx]
        targ = val_target[start_idx:end_idx]
        pc_batch = torch.tensor(pc_batch).float().cuda()
        targ = torch.tensor(targ).float().cuda()
        pred = encoder(pc_batch)
        loss = torch.mean((pred - targ) ** 2)
        val_loss += loss.item()
    val_loss /= num_val_batch
    
    train_losses.append(train_loss)
    test_losses.append(val_loss)
    # print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(ep, train_loss, val_loss))
    pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss})
    
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Val Loss")
plt.legend()
plt.savefig("data/test_dp3_encoder_handle_position_loss.png")
plt.show()
