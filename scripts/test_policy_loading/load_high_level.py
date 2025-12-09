import torch
import os
import numpy as np

def load_multitask_high_level_model(path):
    from omegaconf import OmegaConf
    import json
    ckpt_path = os.path.dirname(path)
    config_path = os.path.join(ckpt_path, "config.json")
    cfg = json.load(open(config_path, "r"))
    cfg = OmegaConf.create(cfg)
    args = cfg
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully load model from: ", path)
    model.eval()
        
    return model, args

def infer_multitask_high_level_model(inputs, goal_prediction_model, cat_embedding=None):
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    with torch.no_grad():
        pred_dict = goal_prediction_model(inputs_, cat_embedding, build_grasp=False, articubot_format=True) 
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, -1)
    
    outputs = outputs.view(B, N, 4, 3)
    
    ### sample an displacement according to the weight
    probabilities = weights  # Must sum to 1
    probabilities = torch.nn.functional.softmax(weights, dim=1)

    # Sample one index based on the probabilities
    sampled_index = torch.argmax(probabilities.squeeze(0))

    displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
    input_point_pos = inputs[:, sampled_index, :] # B, 3
    prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
        
    return prediction


model_path = "/data/yufei/lerobot/data/high-level-ckpt/2025-12-04fine_tune_our_on_sriram_lr_decay/model_15001.pth"
model, _ = load_multitask_high_level_model(model_path)

traj_idx = 0
traj_path = f"/data/yufei/lerobot/data/sriram_plate/traj_{str(traj_idx).zfill(4)}"
t = 0
data_path = os.path.join(traj_path, f"{t}.npz")
data = np.load(data_path, allow_pickle=True)

obj_pcd_np = data['point_cloud'].reshape(-1, 3)
gripper_pcd_np = data['gripper_pcd'].reshape(-1, 3)
goal_gripper_pcd_np = data['goal_gripper_pcd'].reshape(-1, 3)
agent_pos = data['state'].reshape(1, 10)  # (T, 10)

input_pcd = np.concatenate([obj_pcd_np, gripper_pcd_np], axis=0)  # (N, 3)
input_pcd = torch.from_numpy(input_pcd).float().unsqueeze(0)  # B, N, 3
prediction = infer_multitask_high_level_model(
    input_pcd, 
    model,
    cat_embedding=None
)
