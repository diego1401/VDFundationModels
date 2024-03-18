
import torch,os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from opt import config_parser
from dataLoader import ImageLoader, FeatureExtractor
from dataLoader import DinoFeatureExtractor, PATCH_H,PATCH_W
from dataLoader import DiftFeatureExtractor
import cv2

TASK_NAME = "local_cosine_similarity" 

def compute_sift(image):
    # Convert the training image to RGB
    image = image.squeeze(0).permute(1,2,0).numpy()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    #print(gray)
    # Sift geatures
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def save(args, feat_differences, camera_differences):
    save_path = os.path.join("plot_data",TASK_NAME)
    save_path = os.path.join(save_path,args.model_name)
    if args.model_name == "DinoFeatureExtractor":
        save_path = os.path.join(save_path,args.model_size)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path,f"{args.expname}_camera.npy"), 'wb') as f:
        np.save(f, camera_differences)
    with open(os.path.join(save_path,f"{args.expname}_feat.npy"), 'wb') as f:
        np.save(f, feat_differences)

def plot(args,feat_differences,camera_differences):
    save_path = os.path.join("figures",TASK_NAME)
    save_path = os.path.join(save_path,args.model_name)
    if args.model_name == "DinoFeatureExtractor":
        save_path = os.path.join(save_path,args.model_size)
    os.makedirs(save_path, exist_ok=True)
    # Plot the results
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(camera_differences, feat_differences, color='blue', alpha=0.7, edgecolors='black', marker='o')
    plt.xlabel("Camera Similarity", fontsize=12)
    plt.ylabel("Feature Similarity", fontsize=12)
    plt.ylim(0, 1)
    plt.title("Similarity between Camera and Features", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # Ensures the labels fit within the figure area
    plt.savefig(os.path.join(save_path, args.expname +".png"))

    os.makedirs(save_path, exist_ok=True)
    # Plot the results
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.scatter(camera_differences, feat_differences, color='blue', alpha=0.7, edgecolors='black', marker='o')
    plt.xlabel("Camera Similarity", fontsize=12)
    plt.ylabel("Feature Similarity", fontsize=12)
    #plt.ylim(0, 1)
    plt.title("Similarity between Camera and Features", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # Ensures the labels fit within the figure area
    plt.savefig(os.path.join(save_path, args.expname +"_zoomed.png"))

    plt.figure()
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.plot(camera_differences, color='green', marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel("Frame")
    plt.ylabel("Camera Similarity to reference", fontsize=12)
    #plt.ylim(0, 1)
    plt.title("Camera Similarity to reference", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()  # Ensures the labels fit within the figure area
    plt.savefig(os.path.join(save_path,args.expname +"_camera_similarity.png"))

def compare_function(u,v):
    res = np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
    res = np.mean(res)
    #return (res + 1)/2 #normalize to [0,1]
    return res # more intuitive not to normalize

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return torch.tensor([x,y,z])

def camera_extraction_function(pose):
    R = pose[:3,:3]
    T = pose[:3,3].unsqueeze(0)
    # Using euler angles
    #euler_angles = rotation_matrix_to_euler_angles(R).unsqueeze(0)
    #return torch.cat([euler_angles,T]).squeeze(0)
    # Flattening the whole matrix
    #res = pose[:3,:].flatten().unsqueeze(0)
    # Retrieving a directional vector
    directional_vector =  (R @ torch.tensor([0.,0.,1.])).unsqueeze(0)
    res = torch.cat([directional_vector,T]).squeeze(0)
    return res

def compute_global_features(args,feature_extractor:FeatureExtractor,dataset):
    # I. Get reference feature and camera positions
    # feat
    reference_index = 0
    ref_image = dataset[reference_index]["rgbs"].unsqueeze(0)
    get_features = lambda im: feature_extractor.compute_features(im)
    features_ref = get_features(ref_image).cpu().detach().numpy()
    #keypoints_ref,descriptors_ref = compute_sift(ref_image)
    # camera
    get_camera = lambda index: camera_extraction_function(dataset[index]["pose"]).numpy()
    ref_camera = get_camera(reference_index) #transform matrix
    # II. Compare against the rest of the frames
    N = len(dataset)
    feat_differences = []
    camera_differences = []
    
    for shot_id in tqdm(range(1,N)):
        #feat
        current_image = dataset[shot_id]["rgbs"].unsqueeze(0)
        current_feature = get_features(current_image).squeeze(0).cpu().detach().numpy()
        #camera
        current_camera = get_camera(shot_id)
        
        # #keypoints
        # current_keypoints,current_descriptors = compute_sift(current_image)
        # # match keypoints
        # # Create a Brute Force Matcher object.
        # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
        # matches = bf.match(current_descriptors, descriptors_ref)
        # matches = sorted(matches, key = lambda x : x.distance)[:20]
        # # construct list of match pixel coordinates
        # ratio_image_feature = current_feature.shape[0]/current_image.shape[-2]
        # ref_points = np.array([keypoints_ref[m.trainIdx].pt for m in matches])//ratio_image_feature
        # curr_points = np.array([current_keypoints[m.queryIdx].pt for m in matches])//ratio_image_feature
        # selected_feat_ref = np.array([features_ref[int(p[0]),int(p[1])] for p in ref_points])
        # selected_feat_curr = np.array([current_feature[int(p[0]),int(p[1])] for p in curr_points])
        
        # III. Compute the difference
        feat_differences.append(compare_function(features_ref,current_feature))
        camera_differences.append(compare_function(ref_camera,current_camera))
        
        # Change reference to current frame
        
        # keypoints_ref = current_keypoints
        # descriptors_ref = current_descriptors
        features_ref = current_feature
        ref_camera = current_camera
        
    
    # Starting Plot and Save
    feat_differences = np.array(feat_differences)
    camera_differences = np.array(camera_differences)
    # Save
    save(args, feat_differences, camera_differences)
    # Plot
    plot(args,feat_differences,camera_differences)

# Main function to run the program
def main(args, device):
    # I. Load test images and model
    feature_extractor = eval(args.model_name)(device,args)

    data_path = os.path.join(args.project_directory, args.input_dataset)
    dataset = ImageLoader(datadir=data_path,
                          transform=feature_extractor.transform,
                          split="test",
                          img_wh=feature_extractor.img_wh)

    # V. Visualize test data
    compute_global_features(args, feature_extractor, dataset)
    
if __name__ == "__main__":
    args = config_parser()
    assert args is not None
    print(args)
    device = torch.device("cuda:{}".format(args.local_rank) 
                    if torch.cuda.is_available() 
                    else "cpu")
    torch.cuda.set_device(args.local_rank)
    print("Device used is",device)
    main(args,device)
