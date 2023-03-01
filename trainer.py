import argparse
import pickle
import time
from typing import Union

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import FeaturizedDataModule
from engine_abmil import EngineABMIL
from engine_clam import EngineCLAM
from engine_mipcl import EngineMIPCL

# Training settings
parser = argparse.ArgumentParser(
    description="Configurations for WSI training"
)
parser.add_argument(
    '--data_root_dir',
    type=str,
    default="/home/andy/baraslab/projects/panc_cyto/Data/features/256/convnext_gap_L7",
    help='Data directory for extracted pretrained features'
)
parser.add_argument(
    '--kfold_splits_csv_dir',
    type=str,
    default='/home/andy/baraslab/projects/panc_cyto/Data/csv/splits',
    help='CSV directory for the 10-fold stratified splits'
)
parser.add_argument(
    '--max_epoch',
    type=int,
    default=300,
    help='Max # of epochs to train (default: 300)'
)
parser.add_argument(
    '--gpus',
    type=int,
    default=0,
    help='Specify gpu(s) for training (default: gpu:0)'
)
parser.add_argument(
    '--results_dir',
    type=str,
    default=None,
    help='Directory for final results'
)
parser.add_argument(
    '--patience',
    type=int,
    default=50,
    help='Number of epochs for EarlyStopping (default: 50)'
)
parser.add_argument(
    '--model',
    type=str,
    default=None,
    help='Choose which model to run (ABMIL/CLAM/MIPCL)'
)
parser.add_argument(
    '--bag_weight',
    type=float,
    default=None,
    help='Hyperparameter for weighing the two losses (default: None)'
)
parser.add_argument(
    '--in_channels',
    type=int,
    default=None,
    help='Input dim for model'
)
parser.add_argument(
    '--intermediate_dim',
    type=int,
    default=None,
    help='Output dim for encoder'
)
parser.add_argument(
    '--stain_info',
    action='store_action,
    help='Use stain information'
)
parser.add_argument(
    '--dropout',
    action='store_action'
    help='Use dropout'
)
parser.add_argument(
    '--mipcl_alpha',
    type=float,
    default=None,
    help='Use Cosine Similarity for MIPCL instances loss function (default: use InfoNCE)'
)
parser.add_argument(
    '--mipcl_temp',
    type=float,
    default=0.07,
    help='Temperature hyperparameter for MIPCL InfoNCE (default: 0.07)'
)
parser.add_argument(
    '--mipcl_thresh',
    type=float,
    default=0.85,
    help='Grad-CAM softmax probability threshold for MIPCL (default: 0.85)'
)
parser.add_argument(
    '--clam_topk',
    type=int,
    default=8,
    help='Number of top (+) and (-) instances to cluster for CLAM (default: 8)'
)
parser.add_argument(
    '--clam_inst_loss',
    type=str,
    default='svm',
    help='Instance loss for CLAM (default: SmoothTop1SVM)'
)

args = parser.parse_args()

if args.model == 'ABMIL':
    log_dir = f'{args.results_dir}/{args.model}/logs'
    ckpt_dir = f'{args.results_dir}/{args.model}/checkpoints'
elif args.model == 'CLAM':
    log_dir = f'{args.results_dir}/{args.model}/k{args.clam_topk}_il-{args.clam_inst_loss}_bw{args.bag_weight}/logs'
    ckpt_dir = f'{args.results_dir}/{args.model}/k{args.clam_topk}_il-{args.clam_inst_loss}_bw{args.bag_weight}/checkpoints'

if args.model == 'MIPCL' and args.mipcl_alpha is not None:
    log_dir = f'{args.results_dir}/{args.model}/t{args.mipcl_thresh}_cossim{args.mipcl_alpha}/logs'
    ckpt_dir = f'{args.results_dir}/{args.model}/t{args.mipcl_thresh}_cossim{args.mipcl_alpha}/checkpoints'
elif args.model == 'MIPCL' and args.mipcl_temp is not None:
    log_dir = f'{args.results_dir}/{args.model}/t{args.mipcl_thresh}_infonce{args.mipcl_thresh}/logs'
    ckpt_dir = f'{args.results_dir}/{args.model}/t{args.mipcl_thresh}_infonce{args.mipcl_thresh}/checkpoints'


def main(fold_num, args):
    seed_everything(42, workers=True)
    fold_num += 1

    dm = FeaturizedDataModule(
        splits_root_dir=args.kfold_splits_csv_dir,
        features_root_dir=args.data_root_dir,
        fold_num=fold_num,
        batch_size=1,
        num_workers=4
    )

    if args.model == 'ABMIL':
        model = EngineABMIL(
            in_channels=args.in_channels,
            intermediate_dim=args.intermediate_dim,
            n_classes=2,
            stain_info=args.stain_info,
            dropout=args.dropout,
        )
    elif args.model == 'CLAM':
        model = EngineCLAM(
            in_channels=args.in_channels,
            intermediate_dim=args.intermediate_dim,
            n_classes=2,
            stain_info=args.stain_info,
            dropout=args.dropout,
            k_sample=args.clam_topk,
            inst_loss=args.clam_inst_loss,
            bag_weight=args.bag_weight
        )
    elif args.model == 'MIPCL':
        model = EngineMIPCL(
            in_channels=args.in_channels,
            intermediate_dim=args.intermediate_dim,
            n_classes=2,
            stain_info=args.stain_info,
            dropout=args.dropout,
            alpha=args.mipcl_alpha,
            thresh=args.mipcl_thresh,
            temperature=args.mipcl_temp,
            bag_weight=args.bag_weight
        )
    else:
        raise ValueError(
            "Please select one of three implemented models: ABMIL/CLAM/MIPCL")

    tb_logger = TensorBoardLogger(save_dir=log_dir)

    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator='gpu',
        devices=[args.gpus],
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=f'{ckpt_dir}/{fold_num}',
                filename="{epoch}--{avg_val_loss:.4f}",
                save_weights_only=True,
                mode="min",
                monitor="avg_val_loss"
            ),
            EarlyStopping(
                monitor="avg_val_loss",
                min_delta=0.001,
                patience=args.patience,
                mode="min"
            ),
            LearningRateMonitor("epoch"),
        ],
    )

    start = time.time()
    trainer.fit(model, dm)
    print(f'Time elapsed fold {fold_num}: {time.time() - start}s')

    trainer.test(model, dm, ckpt_path="best")
    results = model.test_results
    return results


if __name__ == '__main__':
    settings = {
        'model': args.model,
        'input_dim': args.in_channels,
        'encoder_output_dim': args.intermediate_dim,
        'stain_info': args.stain_info,
        'dropout': args.dropout,
        'bag_weight': args.bag_weight,
        'data_root_dir': args.data_root_dir,
        'csv_splits_dir': args.kfold_splits_csv_dir,
        'max_epoch': args.max_epoch,
        'patience': args.patience,
        'devices': args.gpus,
        'results_dir': args.results_dir,
    }
    if args.mipcl_alpha is not None:
        settings.update(
            {'mipcl_alpha': args.mipcl_alpha}
        )
    if args.mipcl_temp is not None:
        settings.update(
            {'mipcl_temp': args.mipcl_temp}
        )
    if args.model == 'MIPCL' and args.mipcl_thresh is not None:
        settings.update(
            {'mipcl_thresh': args.mipcl_thresh}
        )
    if args.model == 'CLAM' and args.clam_topk is not None:
        settings.update(
            {'clam_topk': args.clam_topk}
        )
    if args.model == 'CLAM' and args.clam_inst_loss is not None:
        settings.update(
            {'clam_inst_loss': args.clam_inst_loss}
        )
    for key, val in settings.items():
        print(f'{key}:  {val}')

    num_kfolds = 10

    results_path = f'{log_dir[:-4]}/stratified_10fold_results.pkl'
    all_results = []
    for i in range(num_kfolds):
        test_results = main(i, args)
        all_results.append(test_results)

    all_results_dict = dict()
    for i in range(num_kfolds):
        fold_num = i + 1
        all_results_dict[f'fold_{fold_num}'] = all_results[i]

    pickle.dump(
        all_results_dict,
        open(results_path, 'wb')
    )
