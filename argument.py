import argparse

def args_parser():
    parser = argparse.ArgumentParser()
##

    # data preprocessing arguments
    parser.add_argument('--HCPdata_dir', type=str, default='/data/camin/yrjang/HCP_data',
                        help='HCP dataset_directory')
    parser.add_argument('--train_size', type=float, default=0.7, 
                        help="ratio of train data")
    parser.add_argument('--val_size', type=float, default=0.15, 
                        help="ratio of validation data")
    parser.add_argument('--test_size', type=float, default=0.15, 
                        help="ratio of test data")
    parser.add_argument('--atlas', type=int, default=300,
                        help='Atlas - (100, 200, 300, 400, MMP)')
    
    # sliding window arguments
    parser.add_argument('--i_win', type=int, default=100, 
                        help="number of timepoints of input window")
    parser.add_argument('--o_win', type=int, default=10, 
                        help="number of timepoints of prediction")
    parser.add_argument('--stride', type=int, default=10, 
                        help="stride size")
    
    
    # model arguments
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="batch_size")
    parser.add_argument('--d_model', type=int, default=256, 
                        help="batch_size")
    parser.add_argument('--nhead', type=int, default=8, 
                        help="batch_size")
    parser.add_argument('--nhid', type=int, default=256, 
                        help="batch_size")
    parser.add_argument('--nlayers', type=int, default=2, 
                        help="batch_size")
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help="batch_size")
    
    parser.add_argument('--lr', type=float, default=0.002, #0.01 ->0.001
                        help='learning rate') 
    parser.add_argument('--epoch', type=int, default=1, #10->50
                        help="number of rounds of training")
    
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)') #sagnet=0.9

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs") #FG=1
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--seed', type=int, default=1, help='random seed')


    parser.add_argument('--dataset-dir', type=str, default='C:/Users/BamiDeep1/Desktop/FLEG0420/dataset',
                        help='Sagnet:: home directory to dataset')
    

    args = parser.parse_args()
    return args

