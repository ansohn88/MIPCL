import argparse
import time
from pathlib import Path
from typing import List

import timm
import torch
from dataset import CytoBagDataset
from torch.nn import functional as F
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader

from utils import collate_features, return_df_from_csv, save_hdf5

device = torch.device(
    "cuda" if torch.cuda.is_available() else torch.device("cpu"))


def compute_features(bag_name: str,
                     bags_dir: str,
                     z_stack: bool,
                     z_level: int,
                     class_path: str,
                     model: timm.models,
                     output_path: str,
                     model_name: str,
                     bsize: int,
                     pretrained: bool = True,
                     verbose: int = 1,
                     #  print_every: int = 20,
                     ):

    dataset = CytoBagDataset(
        filename=bag_name,
        bags_dir=bags_dir,
        z_stack=z_stack,
        z_level=z_level,
        class_path=class_path,
        pretrained=pretrained
    )
    print(f'Number of tiles: {len(dataset)}')

    kwargs = {'num_workers': 8,
              'pin_memory': False}
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=bsize,
        shuffle=False,
        **kwargs,
        collate_fn=collate_features)

    if verbose > 0:
        print(f'Processing {bag_name}: total of {len(data_loader)}')

    with torch.no_grad():
        mode = 'w'
        for iteration, batch in enumerate(data_loader):
            # if iteration % print_every == 0:
            print(
                f'batch {iteration}/{len(data_loader)}, {iteration * bsize} files processed'
            )
            patches = batch["img"].to(device, non_blocking=False)

            if 'convnext' in model_name:
                features = model(patches)[2]
            elif 'resnet' in model_name:
                features = model(patches)[3]
            else:
                features = model.forward_features(patches)

            bsize = features.shape[0]

            # GAP
            features = F.adaptive_avg_pool2d(features, output_size=(1))

            features = features.view(bsize, -1)
            # print(f'FEATURES SHAPE: {features.shape}')
            features = features.cpu().numpy()

            asset_dict = {'features': features,
                          'class_label': batch['label'],
                          'z_level': batch['z_lvls'],
                          'coords': batch['coords']}
            save_hdf5(output_path=output_path,
                      asset_dict=asset_dict,
                      attr_dict=None,
                      mode=mode)
            mode = 'a'

    return output_path


def compute_w_loader(bags_list: list,
                     class_path: str,
                     model_name: str,
                     device_ids: List[int],
                     input_dir: str = None,
                     output_parent: str = None,
                     z_stack: bool = False,
                     z_level: int = 7,
                     ) -> None:
    print('Number of files... ', len(bags_list))

    if z_stack:
        output_parent = f'{output_parent}/convnext_gap'
    else:
        output_parent = f'{output_parent}/convnext_gap_L{z_level}'

    if 'convnext' in model_name:
        print(f'Found ConvNeXt in name... outdir: {output_parent}')
        batch_size = 850
    elif 'resnet' in model_name:
        print(f'Found ResNet in name... outdir: {output_parent}')
        batch_size = 1800
    else:
        raise ValueError("Parent output dir for model_name does not exist!")

    if not Path(output_parent).is_dir():
        Path(output_parent).mkdir()

    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        features_only=True
    )

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DP(model, device_ids=device_ids)
        model.to(f'cuda:{model.device_ids[0]}')

    model.eval()

    total_num_bags = len(bags_list)

    for bag_idx in range(total_num_bags):
        bag_name = str(bags_list[bag_idx])
        print(f'Bag name... {bag_name}')

        start = time.time()

        feat_output_path = f'{output_parent}/{bag_name}'

        if Path(feat_output_path).exists():
            print(f'{feat_output_path} exists already.')
            pass
        else:
            output_file_path = compute_features(
                bag_name=bag_name,
                bags_dir=input_dir,
                class_path=class_path,
                z_stack=z_stack,
                z_level=z_level,
                model=model,
                output_path=feat_output_path,
                model_name=model_name,
                bsize=batch_size
            )
            print(
                f'\nComputing features for {output_file_path} took {time.time()-start}s')


# WSI tessellation settings
parser = argparse.ArgumentParser(
    description="Configurations for feature extract with a pretrained network"
)
parser.add_argument(
    '--tiles_dir',
    type=str,
    default=None,
    help='Directory for tessellated tiles'
)
parser.add_argument(
    '--tile_size',
    type=int,
    default=256,
    help='Size of each tile (default: 256)'
)
parser.add_argument(
    '--out_pdir',
    type=str,
    default=None,
    help='Parent output directory'
)
parser.add_argument(
    '--class_path',
    type=str,
    default=None,
    help='CSV file of filenames and labels'
)
parser.add_argument(
    '--model_name',
    type=str,
    default='convnext_large.fb_in22k_ft_in1k_384',
    help='Model name of pretrained network to use (default: convnext_large.fb_in22k_ft_in1k_384)'
)
parser.add_argument(
    '--z_stack',
    type=bool,
    default=False,
    help='Use all Z-planes (default: False)'
)
parser.add_argument(
    '--z_level',
    type=int,
    default=6,
    help='Extract features from Z-plane (default: 6)'
)
parser.add_argument(
    '--device_ids',
    type=int,
    nargs='+',
    default=None,
    help='Indices of which GPU devices to use'
)

args = parser.parse_args()


if __name__ == '__main__':
    class_csv = args.class_path

    bags_list = return_df_from_csv(class_csv)['original_filename']
    bags_list = [
        f'{(Path(fname).stem)}_bag' for fname in bags_list
    ]

    input_dir = f'{args.tiles_dir}/{args.tile_size}'
    out_dir = f'{args.out_pdir}/{args.tile_size}'

    compute_w_loader(bags_list,
                     class_path=class_csv,
                     input_dir=input_dir,
                     output_parent=out_dir,
                     model_name=args.model_name,
                     z_stack=args.z_stack,
                     z_level=args.z_level,
                     device_ids=args.device_ids)
