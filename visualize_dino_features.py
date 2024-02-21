from cgi import test
from dataLoader.dino_utils import FeatureExtractor
import torch,os, cv2
from tqdm import tqdm
from opt import config_parser
from dataLoader import ImageLoader,load_model,FeatureExtractor, PATCH_H,PATCH_W, FEAT_DIM
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# TODO:
# - Apply only on test data
# - Compute PCA on 3 random images (test images 0,66,134)
# - Make video of the PCA projection
# - Try to do on several scenes (without saving features)


def images_to_video(images,video_name):
    # height, width, _ = images[0].shape
    # video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    # for image in images:
    #     video.write(image)

    # cv2.destroyAllWindows()
    # video.release()
    output_video_path = 'output_video.mp4'

    # Get the height and width of the images
    height, width, _ = images[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 5.0, (width, height))

    # Loop through each image and write it to the video file
    for pca_image in images:
        # Multiply by 255 to convert the image back to the range [0, 255]
        frame = (pca_image * 255).astype(np.uint8)
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video saved at: {output_video_path}")

def visualize_pca(args,array):
    # I. Train PCA on key positions
    # For the NeRF dataset we choose the following test indeces [0,66,134]
    key_indices = [0,33,66]
    features_key_shots = array[key_indices].reshape(3*PATCH_H*PATCH_W,FEAT_DIM)
    #Train PCA to eliminate the background
    pca_remove_bg = PCA(n_components=3)
    pca_remove_bg.fit(features_key_shots)
    pca_features_key_shots = pca_remove_bg.transform(features_key_shots)
    # Remove Background on key shots
    pca_features_key_shots_bg = pca_features_key_shots[:, 0] < 10
    pca_features_key_shots_fg = ~pca_features_key_shots_bg

    # Train PCA to everything but the background
    pca_vis = PCA(n_components=3)
    pca_vis.fit(pca_features_key_shots[pca_features_key_shots_fg]) 
    pca_features_key_shots = pca_vis.transform(pca_features_key_shots[pca_features_key_shots_fg])

    # II. Transform each shot and visualize it

    # Create function to normalize

    def normalize_features(array):
        for i in range(3):
            array[:, i] = (array[:, i] - pca_features_key_shots[:, i].mean()) / (pca_features_key_shots[:, i].std() ** 2) + 0.5

    # Create function to retrieve PCA rgb
            
    def get_pca_rgb(feat):
        pca_feat = pca_remove_bg.transform(feat)
        pca_feat_bg = pca_feat[:, 0] < 10
        pca_feat_fg = ~pca_feat_bg

        pca_feat_rem = pca_vis.transform(pca_feat[pca_feat_fg])
        normalize_features(pca_feat_rem)

        pca_features_rgb = pca_feat.copy()
        pca_features_rgb[pca_feat_bg] = 0
        pca_features_rgb[pca_feat_fg] = pca_feat_rem

        pca_features_rgb = pca_features_rgb.reshape(PATCH_H, PATCH_W, 3)
        return pca_features_rgb

    # Iterate over shots
    figure_path = os.path.join(args.path_to_figures,args.expname)
    os.makedirs(figure_path,exist_ok=True)
    pca_images = []
    upscaled_resolution = (800, 800)  # Adjust to the desired resolution
    for shot_id in range(array.shape[0]):
        current_feature = array[shot_id]
        
        pca_current_feature = np.clip(get_pca_rgb(current_feature),0,1)
        pca_current_feature = cv2.resize(pca_current_feature, upscaled_resolution, interpolation=cv2.INTER_LINEAR)
        pca_current_feature_uint8 = (pca_current_feature * 255).astype(np.uint8)
        # Create the filename for the current image
        filename = os.path.join(figure_path, f'test_pca_features_{shot_id}.png')
        # Save the image using OpenCV
        cv2.imwrite(filename, pca_current_feature_uint8)
        pca_images.append(pca_current_feature)

    #turn to video
    images_to_video(pca_images,video_name="features_visualization.avi")

def main(args,device):
    """
    Script to transform NeRF dataset into sparse view dataset of DinoV2 features.
    """
    #I. Load test images and model
    data_path = os.path.join(args.project_directory, args.input_dataset)
    test_dataset = ImageLoader(datadir=data_path,split="test")
    print("Loading Model...")
    dinov2_model = load_model()
    dinov2_model.to(device)
    #II. Transform and Save Images into features
    print("Getting DiNO features")
    path_to_features = os.path.join(args.feature_dataset,args.expname)
    path = os.path.join(args.path_to_hdd, path_to_features)
    os.makedirs(path,exist_ok=True)
    feature_extractor = FeatureExtractor(dinov2_model,args.batch_size,device)
    test_features = feature_extractor.get_dino_features(test_dataset,os.path.join(path,"test"))
    #V. Visualize test data
    visualize_pca(args,test_features)
    
if __name__ == "__main__":
    args = config_parser()
    assert args is not None
    print(args)
    device = torch.device("cuda:{}".format(args.local_rank) 
                    if torch.cuda.is_available() 
                    else "cpu")
    main(args,device)