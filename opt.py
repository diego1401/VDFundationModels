import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # General information
    parser.add_argument("--project_directory", type=str,
                        help='experiment name')
    
    # Incoming Dataset Information
    parser.add_argument("--input_dataset", type=str,
                        help='experiment name')
    
    # Output Dataset Information
    parser.add_argument("--feature_dataset", type=str,
                        help='experiment name') 