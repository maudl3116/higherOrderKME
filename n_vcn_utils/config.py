import argparse
import numpy as np
from utils import rand_int

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./50.)

parser.add_argument('--n_ball', type=int, default=0, help="option for ablating on the number of balls")
parser.add_argument('--rel_type', type=int, default=1, help="type of the relation when there is one") # added this
parser.add_argument('--rel_attr', type=float, default=0, help="attribute of the relation") # added this

parser.add_argument('--stage', default='kp', help='kp|dy')
parser.add_argument('--dataf', default='data')

'''
train
'''
parser.add_argument('--random_seed', type=int, default=1024)

parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
parser.add_argument('--train_valid_ratio', type=float, default=0.5, help="percentage of training data")

parser.add_argument('--height_raw', type=int, default=0)
parser.add_argument('--width_raw', type=int, default=0)
parser.add_argument('--height', type=int, default=0)
parser.add_argument('--width', type=int, default=0)
parser.add_argument('--scale_size', type=int, default=0)
parser.add_argument('--crop_size', type=int, default=0)

parser.add_argument('--eval', type=int, default=0)

parser.add_argument('--n_rollout', type=int, default=5, help='number of rollout steps for training')

parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--node_attr_dim', type=int, default=0)
parser.add_argument('--edge_attr_dim', type=int, default=0)
parser.add_argument('--edge_type_num', type=int, default=0)
parser.add_argument('--edge_st_idx', type=int, default=0, help="whether to exclude the first edge type")
parser.add_argument('--edge_share', type=int, default=0,
                    help="whether forcing the info being the same for both directions")
parser.add_argument('--video', type=int, default=0,
                    help="whether to generate a video")
parser.add_argument('--image', type=int, default=0,
                    help="whether to generate images")
parser.add_argument('--draw_edge', type=int, default=0,
                    help="whether to draw edges on video, image")


'''
eval
'''
parser.add_argument('--evalf', default='eval')

parser.add_argument('--eval_set', default='valid', help='train|valid')
parser.add_argument('--eval_st_idx', type=int, default=0)
parser.add_argument('--eval_ed_idx', type=int, default=0)

parser.add_argument('--vis_edge', type=int, default=1)
parser.add_argument('--store_demo', type=int, default=1)
parser.add_argument('--store_result', type=int, default=0)
parser.add_argument('--store_st_idx', type=int, default=0)
parser.add_argument('--store_ed_idx', type=int, default=0)

'''
model
'''
# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)



def gen_args():
    args = parser.parse_args()

    if args.env == 'Ball':
        args.data_names = ['attrs', 'states', 'actions', 'rels']

        args.frame_offset = 1
        args.train_valid_ratio = 0.01

        # radius
        args.attr_dim = 1
        # x, y, xdot, ydot
        args.state_dim = 4
        # ddx, ddy
        args.action_dim = 2
        # none, spring, rod
        args.relation_dim = 3

        # size of the latent causal graph
        args.node_attr_dim = 0
        args.edge_attr_dim = 1
        args.edge_type_num = 3

        # generate random relations                                 # added this
        nb_edges = args.n_ball * (args.n_ball - 1) // 2
        load_rels = np.zeros((nb_edges,2))
        # the balls that are connected have the same type of relation (spring or string) and the same attribute
        for i in range(nb_edges):
            if rand_int(0, 2)==1:
                load_rels[i,0] = args.rel_type
                load_rels[i,1] = args.rel_attr  
        args.load_rels = load_rels

        args.height_raw = 110
        args.width_raw = 110
        args.height = 64
        args.width = 64
        args.scale_size = 64
        args.crop_size = 64

        args.lim = [-1., 1., -1., 1.]

    elif args.env == 'Cloth':
        args.data_names = ['states', 'actions', 'scene_params']

        args.n_rollout = 2000
        if args.stage == 'dy':
            args.frame_offset = 5
        else:
            args.frame_offset = 1
        args.time_step = 300 // args.frame_offset
        args.train_valid_ratio = 0.9

        # x, y, z, xdot, ydot, zdot
        args.state_dim = 6
        # x, y, z, dx, dy, dz
        args.action_dim = 6

        # size of the latent causal graph
        args.node_attr_dim = 0
        args.edge_attr_dim = 1
        args.edge_type_num = 2

        args.height_raw = 400
        args.width_raw = 400
        args.height = 64
        args.width = 64
        args.scale_size = 64
        args.crop_size = 64

        args.lim = [-1., 1., -1., 1.]

    else:
        raise AssertionError("Unsupported env %s" % args.env)


    # path to data
    args.dataf = 'data/' + args.dataf + '_' + args.env

    return args

