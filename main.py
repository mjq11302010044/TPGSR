import yaml
import sys
import argparse
import os
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR


def main(config, args, opt_TPG):
    Mission = TextSR(config, args, opt_TPG)

    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='tsrn_tl_wmask', choices=['tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn', 'tsrn_tl_wmask', 'tsrn_tl_cascade', 'srcnn_tl', 'srresnet_tl', 'rdn_tl', 'vdsr_tl'])
    parser.add_argument('--go_test', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='../hard_space1/mjq/TextZoom/test/medium/', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--ic15sr', action='store_true', default=False, help='use IC15SR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--stu_iter', type=int, default=1, help='Default is set to 1, must be used with --arch=tsrn_tl_cascade')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN"])
    parser.add_argument('--sr_share', action='store_true', default=False)
    parser.add_argument('--tpg_share', action='store_true', default=False)
    parser.add_argument('--use_label', action='store_true', default=False)
    parser.add_argument('--use_distill', action='store_true', default=False)
    parser.add_argument('--ssim_loss', action='store_true', default=False)
    parser.add_argument('--random_reso', action='store_true', default=False)
    parser.add_argument('--tpg', type=str, default="CRNN", choices=['CRNN', 'OPT'])
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)


    parser_TPG = argparse.ArgumentParser()

    opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

    opt["num_class"] = len(opt['character'])

    opt = EasyDict(opt)
    main(config, args, opt_TPG=opt)
