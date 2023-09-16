import torch
from models.detr import build_model
from data_opt.load_data import PredictPicture
from transforms import _get_transform
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='checkpoint/checkpoint.pth',
                        help='path where model saved')
parser.add_argument('--prediction_data', default="paper/00001448.jpg",
                        help='path where picture to predict .')
args = parser.parse_args()

def predict_picture():
    model_parameter=torch.load(args.checkpoint_path)
    args_=model_parameter["args"]
    model= build_model(args_)
    model.load_state_dict(model_parameter["model"])

    pred=PredictPicture(model,_get_transform(),args_)

    predict_=pred(args.prediction_data)
    for i in predict_:
        print(i)
    return predict_


if __name__=="__main__":
    predict_picture()