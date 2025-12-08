from manipulation.utils import build_up_env
from manipulation.utils import load_env, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
import numpy as np
import pybullet as p
from manipulation.gpt_reward_api import get_link_id_from_name

def get_filter_obs_pcd_plane(task_config_path, solution_path, first_step, obj_name, handle_length_offset=0.2):
	env, _ = build_up_env(
		task_config=task_config_path,
		solution_path=solution_path,
		task_name=first_step.replace(" ", "_"),
		restore_state_file=None,
		render=False,
		randomize=False,
		obj_id=0,
	)

	obj_id = env.urdf_ids[obj_name]
	base_pos, base_orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)

	# reset the object orientation to [0, 0, 0, 1]
	new_orn = [0, 0, 0, 1]
	p.resetBasePositionAndOrientation(obj_id, base_pos, new_orn, physicsClientId=env.id)

	# close the door
	link_id = get_link_id_from_name(env, obj_name, "link_0")
	p.resetJointState(obj_id, link_id, 0, physicsClientId=env.id)

	# get the bounding box of the object
	min_aabb, max_aabb = env.get_aabb(obj_id)

	# the plane before rotation
	# x < min_aabb[0] + handle_length_offset
	plane = np.array([-1, 0, 0, min_aabb[0] + handle_length_offset])

	# rotate the plane to the object orientation
	euler = p.getEulerFromQuaternion(base_orn)
	z_rot = euler[2]

	# rotate the plane
	R_z = np.array([[np.cos(z_rot), -np.sin(z_rot), 0], [np.sin(z_rot), np.cos(z_rot), 0], [0, 0, 1]])
	normal = np.array([-1, 0, 0])
	point_on_plane = np.array([min_aabb[0] + handle_length_offset, 0, 0])
	translate_to_origin = np.array([-base_pos[0], -base_pos[1], -base_pos[2]])
	rotated_normal = R_z @ normal
	new_a, new_b, new_c = rotated_normal
	rotated_point_on_plane = point_on_plane + translate_to_origin
	rotated_point_on_plane = R_z @ rotated_point_on_plane
	rotated_point_on_plane = rotated_point_on_plane - translate_to_origin
	new_d = -new_a * rotated_point_on_plane[0] - new_b * rotated_point_on_plane[1] - new_c * rotated_point_on_plane[2]
	plane = np.array([new_a, new_b, new_c, new_d])

	env.close()

	return plane



if __name__ == "__main__":
	task_config_path = "/data/ziyuw2/Projects/RoboGen-sim2real/data/diverse_objects_2/open_the_door_45633/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/1119-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-11-18-12-09-30/task_config.yaml"
	solution_path = "data/diverse_objects_2/open_the_door_45633/task_open_the_door_of_the_storagefurniture_by_its_handle"
	first_step = "grasp_the_handle_of_the_storage_furniture_door"
	obj_name = "storagefurniture"
	get_filter_obs_pcd_plane(task_config_path, solution_path, first_step, obj_name)




	

