
import torch,os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from opt import config_parser
from dataLoader import ImageLoader, FeatureExtractor
from dataLoader import DinoFeatureExtractor, PATCH_H,PATCH_W
from dataLoader import DiftFeatureExtractor


TASK_NAME = "cross_similarity"
AVAILABLE_MODELS = ["DinoFeatureExtractor","DiftFeatureExtractor"]
#AVAILABLE_MODELS = ["DiftFeatureExtractor"]
SCENES = ["chair","drums","ficus","hotdog","lego","materials","mic","ship"]
PROMPTS = {"chair":"chair","drums":"drums","ficus":"ficus","hotdog":"hotdog","lego":"lego truck","materials":"shiny spheres","mic":"mic","ship":"ship"}
PATH = "data/nerf_synthetic/"
N_FRAMES_TO_COMPARE = 5

def save(model_name, cross_similarity_dictionary):
    save_path = os.path.join("plot_data",TASK_NAME)
    save_path = os.path.join(save_path,model_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path,f"{model_name}.pkl"), 'wb') as f:
        pickle.dump(cross_similarity_dictionary, f)
        
def compare_function(u,v):
    res = np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
    res = np.mean(res)
    return res # more intuitive not to normalize

def compute_similarity(model_name, feature_extractor:FeatureExtractor,dataset,dataset_to_compare,scene_name,scene_to_compare):
    '''
    Given a model and a shape return mean similarity to all remaining shapes
    '''
    similarities = []
    random_idx = np.random.choice(len(dataset),N_FRAMES_TO_COMPARE,replace=False)
    random_idx_compare = np.random.choice(len(dataset_to_compare),N_FRAMES_TO_COMPARE,replace=False)
    
    if model_name == "DiftFeatureExtractor":
        get_feature = lambda img: feature_extractor.compute_features(img,PROMPTS[scene_name]).squeeze(0).cpu().detach().numpy()
        get_feature_compare = lambda img: feature_extractor.compute_features(img,PROMPTS[scene_to_compare]).squeeze(0).cpu().detach().numpy()
    else:
        get_feature = lambda img: feature_extractor.compute_features(img).squeeze(0).cpu().detach().numpy()
        get_feature_compare = lambda img: feature_extractor.compute_features(img).squeeze(0).cpu().detach().numpy()
        
    for i in random_idx:
        img = dataset[i]["rgbs"].unsqueeze(0)
        feature = get_feature(img)
        for j in random_idx_compare:
            img_to_compare = dataset_to_compare[j]["rgbs"].unsqueeze(0)
            feature_to_compare = get_feature_compare(img_to_compare)
            similarities.append(compare_function(feature,feature_to_compare))
    return np.mean(similarities)

def compute_cross_shape_similarity(args,feature_extractor:FeatureExtractor, model_name):
    '''
    Given a model and a shape return mean similarity to all remaining shapes
    '''
    cross_similarities_dict = {}
    for scene_name in tqdm(SCENES):
        # I. Choose a scene
        dataset = ImageLoader(datadir=os.path.join(args.project_directory, os.path.join(PATH,scene_name)),
                            transform=feature_extractor.transform,
                            split="train",
                            img_wh=feature_extractor.img_wh)
        remaining_shapes = [name for name in SCENES if name != scene_name]
        cross_similarities = []
        for scene_to_compare in remaining_shapes:
            # II. Choose a scene to compare
            dataset_to_compare = ImageLoader(datadir=os.path.join(args.project_directory, os.path.join(PATH,scene_to_compare)),
                                            transform=feature_extractor.transform,
                                            split="train",
                                            img_wh=feature_extractor.img_wh)
            # III. Compute similarity
            cross_similarities.append(compute_similarity(model_name,feature_extractor,dataset,dataset_to_compare,scene_name,scene_to_compare))
        cross_similarities_dict[scene_name] = np.mean(cross_similarities)
        print(f"Mean similarity for {scene_name} is {cross_similarities_dict[scene_name]}")
    return cross_similarities_dict
    

# Main function to run the program
def main(args, device):
    # I. Load test images and model
    for model_name in AVAILABLE_MODELS:
        if model_name == "DinoFeatureExtractor":
            args.model_size = "big"
        feature_extractor = eval(model_name)(device,args)
        cross_sim = compute_cross_shape_similarity(args, feature_extractor, model_name)
        # Save
        save(model_name, cross_sim)
    
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
