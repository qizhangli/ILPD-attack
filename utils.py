import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

from dataset import SelectedImagenet


def get_transforms(data_config, source=True):
    transforms = timm.data.transforms_factory.create_transform(
                        input_size = data_config['input_size'],
                        interpolation = data_config['interpolation'],
                        mean=(0,0,0),
                        std=(1,1,1),
                        crop_pct=data_config['crop_pct'] if not source else 1.,
                        tf_preprocessing=False,
                    )
    if not source:
        transforms.transforms = transforms.transforms[:-2]
    return transforms

def build_dataset(args, data_config):
    img_transform = get_transforms(data_config)
    dataset = SelectedImagenet(imagenet_val_dir=args.data_dir,
                               selected_images_csv=args.data_info_dir,
                               transform=img_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory = True, num_workers=4)
    return data_loader
    
def build_model(model_name):
    model = eval("timm.models.{}(pretrained=True)".format(model_name))
    data_config = model.pretrained_cfg
    model = nn.Sequential(T.Normalize(data_config["mean"], 
                                      data_config["std"]), 
                          model)
    model = nn.DataParallel(model)
    model.eval()
    model.cuda()
    return model, data_config
