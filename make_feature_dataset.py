from cgi import test
from dataLoader.dino_utils import FeatureExtractor
import torch,os, cv2
from tqdm import tqdm
from opt import config_parser
from dataLoader import ImageLoader,load_model,FeatureExtractor, PATCH_H,PATCH_W, FEAT_DIM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def images_to_video(images,video_name):
    height, width, _ = images[0].shape
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def visualize(args,array):
    #Down Sample data
    features = array.reshape(4 * PATCH_H * PATCH_W, FEAT_DIM)
    
    #Apply PCA to eliminate the background
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    # segment using the first component
    pca_features_bg = pca_features[:, 0] < 10
    pca_features_fg = ~pca_features_bg

    # Apply PCA to everything but the background
    pca.fit(features[pca_features_fg]) 
    pca_features_rem = pca.transform(features[pca_features_fg])
    for i in range(3):
        # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
        # transform using mean and std, I personally found this transformation gives a better visualization
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_rem

    pca_features_rgb = pca_features_rgb.reshape(4, PATCH_H, PATCH_W, 3)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(pca_features_rgb[i][..., ::-1])
    plt.savefig('features.png')
    plt.show()
    plt.close()
    #turn to video
    #images_to_video(feature_images,video_name="features_visualization.avi")

def main(args,device):
    """
    Script to transform NeRF dataset into sparse view dataset of DinoV2 features.
    """
    #I. Load images and model
    data_path = os.path.join(args.project_directory, args.input_dataset)
    train_dataset = ImageLoader(datadir=data_path,split="train")
    test_dataset = ImageLoader(datadir=data_path,split="test")
    val_dataset = ImageLoader(datadir=data_path,split="val")
    print("Loading Model...")
    dinov2_model = load_model()
    dinov2_model.to(device)
    #II. Transform and Save Images into features
    print("Getting DiNO features")
    path_to_features = os.path.join(args.feature_dataset,args.expaname)
    path = os.path.join(args.path_to_hdd, path_to_features)
    os.makedirs(path,exist_ok=True)
    feature_extractor = FeatureExtractor(dinov2_model,args.batch_size,device)
    train_features = feature_extractor.get_dino_features(train_dataset,os.path.join(path,"train"),save=True)
    val_features = feature_extractor.get_dino_features(val_dataset,os.path.join(path,"val"),save=True)
    test_features = feature_extractor.get_dino_features(test_dataset,os.path.join(path,"test"),save=True)
    #V. Visualize test data
    visualize(args,test_features[:4])
    
if __name__ == "__main__":
    args = config_parser()
    assert args is not None
    print(args)
    device = torch.device("cuda:{}".format(args.local_rank) 
                    if torch.cuda.is_available() 
                    else "cpu")
    main(args,device)