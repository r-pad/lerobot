import numpy as np
#Uncomment these lines just to run main from this script
#from aug_rotation_z import rotationZ
#from aug_translation_xy import TranslationXY
import pickle
#import matplotlib.pyplot as plt
class RandomApplyNumpy():
    def __init__(self,transforms_and_probs):
        self.transforms_and_probs = transforms_and_probs
        
    def __call__(self, data):
        random_number = np.random.rand()
        for transform_prob in self.transforms_and_probs:
            #print(transform_prob[1])
            assert transform_prob[1] <= 1.0 and transform_prob[1] >= 0.0, "Augmentation probabilities much be less than 1 and greater than 0"
            if random_number < transform_prob[1]:
                data =  transform_prob[0](data)

        return data
    

if __name__ == "__main__":
    mean_x = 0
    std_x = 2
    mean_y = 0
    std_y = 2
    trans_x = TranslationXY(mean_x, mean_y, std_x, std_y, True, False)
    trans_y = TranslationXY(mean_x, mean_y, std_x, std_y, False, True)
    mean_angle_z = 0 
    std_rot_z = 30
    rot_z = rotationZ(mean_angle_z, std_rot_z)
    #probs = [0.4, 0.6, 0.3]
    transforms_and_probs = [[trans_x,0], [trans_y, 0], [rot_z, 1]]
    rand_apply = RandomApplyNumpy(transforms_and_probs)

    data = pickle.load(open("/scratch/chialiang/dp3_demo/0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action/2024-05-11-00-24-52/0.pkl", 'rb'))
    data = rand_apply(data)
    with open('transforms_z_rot_1_array.pkl', 'wb') as f:
        pickle.dump(data, f)
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first set of points
    ax.scatter(data["point_cloud"][0,:,0], data["point_cloud"][0,:,1], data["point_cloud"][0,:,2], color='blue', label='Set 1', marker='o', alpha=0.5)
    ax.scatter(data["gripper_pcd"][0,:,0], data["gripper_pcd"][0,:,1], data["gripper_pcd"][0,:,2], color='red', label='Set 2', marker='o')
    ax.scatter(data["goal_gripper_pcd"][0,:,0], data["goal_gripper_pcd"][0,:,1], data["goal_gripper_pcd"][0,:,2], color='green', label='Set 3', marker='o')
    # Plot the second set of points
    #ax.scatter(x2, y2, z2, color='red', label='Set 2', marker='x')

    # Add titles and labels
    ax.set_title('3D Scatter Plot of Two Sets of Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Add a legend
    ax.legend()

    # Save the plot to a file
    plt.savefig('Augmentation_Check_only_z_rot_1.png', format='png')






