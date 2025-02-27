import argparse

def args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--num_frames_input',      default=12                       )
    parser.add_argument('--num_frames_output',     default=12                       )
    parser.add_argument('--frames_per_subsample',  default=24                       )
    parser.add_argument('--next_step',             default=2                        )
    parser.add_argument('--train_data_dir',        default='./data/thetao_train.nc' )
    parser.add_argument('--test_data_dir',         default='./data/thetao_test.nc'  )

    # model
    parser.add_argument('--model_name',  default='SVRNN',         )
    parser.add_argument('--in_shape',    default=[24, 1, 72, 72], )
    parser.add_argument('--num_layers',  default=1,               )
    parser.add_argument('--num_hidden',  default=[5],             )
    parser.add_argument('--filter_size', default=3,               )
    parser.add_argument('--stride',      default=1,               )
    parser.add_argument('--layer_norm',  default=1,               )

    # train
    parser.add_argument('--model_save_path',    default='./model_save',      )
    parser.add_argument('--check_save_path',    default='./Checkpoint',      )
    parser.add_argument('--check_seq_len',      default=20,                  )
    parser.add_argument('--epochs',             default=200,                 )
    parser.add_argument('--batch_size',         default=2,                   )
    parser.add_argument('--lr',                 default=0.001,               )
    parser.add_argument('--num_workers',        default=8                    )

    args = parser.parse_args()
    return args
