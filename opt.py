import argparse

def get_opts() :
    parser = argparse.ArgumentParser() 

    parser.add_argument('--num_epochs', type=int, default = 10,
                        help='the num  of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='the learning rate for model')
    parser.add_argument('--num_epochs', type=int, default = 10,
                        help='the num  of epochs')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='the saved dir for out model')
    parser.add_argument('--ues_vld', type=bool, default = True,
                        help='use the VisualDL')

    return parser.parse_args()

