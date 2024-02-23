import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # General information
    parser.add_argument("--project_directory", type=str,
                        help='path of current project',
                        default='')
    parser.add_argument("--local_rank", type=int,
                        help='gpu id to use',
                        default=0) 
    
    # Incoming Dataset Information
    parser.add_argument("--input_dataset", type=str,
                        help='path to folder containing data')
    
    # Output Dataset Information
    parser.add_argument("--feature_dataset", type=str,
                        help='path to store features') 
    parser.add_argument("--expname", type=str,
                        help='Name of Experiment') 
    parser.add_argument("--path_to_hdd", type=str,
                        help='path to bigger storage') 
    parser.add_argument("--path_to_figures", type=str,
                        help='path to store figures') 
    
    # Machine Learning Parameters
    parser.add_argument("--batch_size", type=int,
                        help='Size of batches used for inference') 
    parser.add_argument("--model_size", type=str,
                        help='Size of the model used',
                        default='small') 

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()