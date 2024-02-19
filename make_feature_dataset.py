###
# TODO: - Make a dataset of 100 “images” of the features of a scene.
#       - Make function to visualize result → PCA to 3D and then draw.
###


import torch,os, cv2
from opt import config_parser
from dataLoader import ImageLoader,POSSIBLE_SPLITS

def load_model():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    return dinov2_vits14

def apply_dino_features(dataset,model):
    all_features = []
    for samples in dataset:
        image_batch = samples["rgbs"]
        #Apply model
        features_batch = model(image_batch)
        all_features.append(features_batch)
    return torch.stack(all_features,dim=0).detach().cpu()

def save_features(array, prefix):
    N = array.shasave_featurespe[0]
    assert prefix in POSSIBLE_SPLITS
    for i in range(N):
        feature = array[i]
        feature_name = f"{prefix}_feature_{i}.pt"
        torch.save(array,f=feature_name)

def PCA_down_sampling(array,k=3):
    #Scale Data
    # Calculate the mean and standard deviation for standardization
    mean = torch.mean(array, dim=0)
    std_dev = torch.std(array, dim=0)

    # Perform standardization by subtracting the mean and dividing by standard deviation
    scaled_data = (array - mean) / std_dev
    #apply PCA
    _, _, V = torch.svd(scaled_data)
    principal_components = V[:, :k]
    return principal_components

def images_to_video(images,video_name):
    height, width, _ = images[0].shape
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def visualize(array):
    #Down Sample data
    feature_images = PCA_down_sampling(array).numpy()
    #turn to video
    images_to_video(feature_images,video_name="features_visualization.avi")

def main(args):
    """
    Script to transform NeRF dataset into sparse view dataset of DinoV2 features.
    """

    #I. Create Directory to save dataset
    path = os.path.join(args.project_directory, args.feature_dataset) 
    os.mkdir(path) 
    print("Directory '% s' created" % path) 
    #II. Load images and model
    data_path = os.path.join(args.project_directory, args.input_dataset)
    train_dataset = ImageLoader(datadir=data_path,split="train")
    test_dataset = ImageLoader(datadir=data_path,split="test")
    val_dataset = ImageLoader(datadir=data_path,split="val")
    dinov2_model = load_model()
    #III. Transform Images into features
    train_features = apply_dino_features(train_dataset,dinov2_model)
    test_features = apply_dino_features(test_dataset,dinov2_model)
    val_features = apply_dino_features(val_dataset,dinov2_model)
    #IV. Save features
    save_features(train_features,prefix="train")
    save_features(test_features,prefix="test")
    save_features(val_features,prefix="val")
    #V. Visualize test data
    visualize(test_features)
    
if __name__ == "__main__":
    args = config_parser()
    main(args)