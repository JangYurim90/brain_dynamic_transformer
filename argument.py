import argparse

def args_parser():
    parser = argparse.ArgumentParser()
##

    parser.add_argument('--mode', type=str, default='train', help='train or test')
    
    # data preprocessing arguments
    parser.add_argument('--HCPdata_dir', type=str, default='/data/camin/yrjang/HCP_data',
                        help='HCP dataset_directory')
    parser.add_argument('--loss_dir', type=str, default='/data/camin/yrjang/Brain_network_dynamics/Dynamics_RNN/train_result',
                        help='HCP dataset_directory')
    
    parser.add_argument('--train_size', type=float, default=0.7, 
                        help="ratio of train data")
    parser.add_argument('--val_size', type=float, default=0.15, 
                        help="ratio of validation data")
    parser.add_argument('--test_size', type=float, default=0.15, 
                        help="ratio of test data")
    parser.add_argument('--atlas', type=str, default='300',
                        help='Atlas - (100, 200, 300, 400, MMP)')
    
    # sliding window arguments
    parser.add_argument('--i_win', type=int, default=100, 
                        help="number of timepoints of input window")
    parser.add_argument('--o_win', type=int, default=10, 
                        help="number of timepoints of prediction")
    parser.add_argument('--stride', type=int, default=10, 
                        help="stride size")
    
    
    # model arguments
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="batch_size")
    parser.add_argument('--d_model', type=int, default=64, 
                        help='Internal dimension of transformer embeddings')
    parser.add_argument('--nhead', type=int, default=4, 
                        help='Number of multi-headed attention heads')
    parser.add_argument('--nhid', type=int, default=256, 
                        help="number of hidden units in the feedforward network")
    parser.add_argument('--feature_dim', type=int, default=314,
                        help = "number of features")
    
    parser.add_argument('--nlayers', type=int, default=2, 
                       help='Number of transformer encoder layers (blocks)')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimension of dense feedforward part of transformer layer')
    parser.add_argument('--dropout', type=float, default=0.01, 
                        help='Dropout applied to most transformer encoder layers')
    
    parser.add_argument('--lr', type=float, default=0.005, #0.01 ->0.001
                        help='learning rate') 
    parser.add_argument('--epoch', type=int, default=100, #10->50
                        help="number of rounds of training")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)') #sagnet=0.9

    # model arguments
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
    parser.add_argument('--l2_reg, type=bool', default=True)

    # other arguments
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    

    args = parser.parse_args()
    return args

