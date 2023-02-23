import os
import pickle
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image


class Visualization:
    """
    dict_keys(['test_loss', 
                'predictions', 
                'probabilities', 
                'targets',
                'test_recall',
                'test_precision',
                'test_f1_score',
                'auprc',
                'confmat'])
    """

    def __init__(
        self,
        tiles_dir: str,
        top_tiles_outdir: str,
        results_fp: str,
        metric: str = 'test_recall',
        best_or_worst: str = 'best',
        # model: str = 'clam'
    ) -> None:

        self.results_fp = results_fp

        self.results = pickle.load(open(results_fp, 'rb'))
        self.tiles_dir = tiles_dir
        self.top_tiles_outdir = top_tiles_outdir

        self.metric = metric
        self.best_or_worst = best_or_worst

        self.fold_results_all = self.get_all_fold_results(
            metric=metric,
            best_or_worst=best_or_worst
        )
        self.fold_num = self.fold_results_all['fold_num']
        self.fold_metric_results = self.results[f'fold_{self.fold_num}']

    def get_top_tiles(self,
                      save_tiles: bool,
                      fold_num: Optional[int],
                      id_only: bool) -> Optional[dict]:

        all_preds = self.get_all_preds(save_file=False)
        if fold_num is not None:
            fnames_topk_ids = self.results[f'fold_{fold_num}']['fnames_topk_ids']
            preds = all_preds[int(fold_num-1)][2]
        else:
            fnames_topk_ids = self.fold_metric_results['fnames_topk_ids']
            preds = all_preds[self.fold_num][2]

        tiles_w_ids = dict()
        for filename, _ in fnames_topk_ids.items():
            fname_results = fnames_topk_ids[filename]

            if preds == 1:
                fname_top_ids = fname_results['top_fg_id']['pos_idx'].detach(
                ).cpu().numpy()
            elif preds == 0:
                fname_top_ids = fname_results['top_fg_id']['neg_idx'].detach(
                ).cpu().numpy()
            else:
                raise ValueError("Prediction is not 0 or 1...!")

            tiles_path = f'{self.tiles_dir}/{filename[:-12]}.h5'
            fname_tiles = self.load_tiles(
                filepath=tiles_path
            )

            top_tiles = fname_tiles[fname_top_ids, ...]

            if save_tiles:
                out_dir = f'{self.top_tiles_outdir}/fold_{fold_num}/{filename[:-16]}'
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)

                for i, pid in enumerate(fname_top_ids):
                    tile = Image.fromarray(
                        fname_tiles[pid, ...]
                    )
                    tile.save(
                        f'{out_dir}/tile_p_{i}_idx_{pid}.png'
                    )
            else:
                if id_only:
                    tiles_w_ids[f'{filename[:-16]}'] = {
                        'top_p': fname_top_ids,
                    }
                else:
                    tiles_w_ids[f'{filename[:-16]}'] = {
                        'top_p': [top_tiles, fname_top_ids],
                    }

        return tiles_w_ids

    def get_top_tiles_single_case(self,
                                  filename: str,
                                  save_tiles: bool,
                                  fold_num: Optional[int],
                                  id_only: bool) -> Optional[dict]:
        all_preds = self.get_all_preds(save_file=False)
        if fold_num is not None:
            fnames_topk_ids = self.results[f'fold_{fold_num}']['fnames_topk_ids']
            preds = all_preds[int(fold_num-1)][2]
        else:
            fnames_topk_ids = self.fold_metric_results['fnames_topk_ids']
            preds = all_preds[self.fold_num][2]

        tiles_w_ids = dict()

        fname_results = fnames_topk_ids[filename]

        if preds == 1:
            fname_top_ids = fname_results['top_fg_id']['pos_idx'].detach(
            ).cpu().numpy()
        elif preds == 0:
            fname_top_ids = fname_results['top_fg_id']['neg_idx'].detach(
            ).cpu().numpy()
        else:
            raise ValueError("Prediction is not 0 or 1...!")

        tiles_path = f'{self.tiles_dir}/{filename[:-12]}.h5'

        fname_tiles = self.load_tiles(
            filepath=tiles_path
        )

        top_tiles = fname_tiles[fname_top_ids, ...]

        if save_tiles:
            out_dir = f'{self.top_tiles_outdir}/fold_{fold_num}/{filename[:-16]}'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            for i, pid in enumerate(fname_top_ids):
                tile = Image.fromarray(
                    fname_tiles[pid, ...]
                )
                tile.save(
                    f'{out_dir}/tile_p_{i}_idx_{pid}.png'
                )
        else:
            if id_only:
                tiles_w_ids[f'{filename[:-16]}'] = {
                    'top_p': fname_top_ids,
                }
            else:
                tiles_w_ids[f'{filename[:-16]}'] = {
                    'top_p': [top_tiles, fname_top_ids],
                }

        return tiles_w_ids

    def get_attn_vals(self) -> dict:

        attn_w_tiles = dict()
        fnames_topk_ids = self.fold_metric_results['fnames_topk_ids']
        print(self.fold_metric_results)
        for filename, _ in fnames_topk_ids.items():
            fname_results = self.fold_metric_results[f'{filename}']
            original_fname = filename[:-16]

            tiles_w_ids = self.get_top_tiles(save_tiles=False)[original_fname]

            attn_w_tiles[original_fname] = {
                'A_raw': fname_results['A_raw'].detach().cpu().numpy(),
                'tiles_w_ids': tiles_w_ids
            }

        return attn_w_tiles

    def get_confmat_metrics(self,
                            return_metric: bool,
                            ) -> dict:
        confusion_mtxs = self.fold_results_all['conf_mats']
        confmat_metrics = dict()

        if return_metric:
            conf_mat = confusion_mtxs[self.fold_num]
            metrics = self.recall_precision_from_confmat(conf_mat)
            confmat_metrics[str(self.fold_num)] = metrics
        else:
            for idx in range(len(confusion_mtxs)):
                conf_mat = confusion_mtxs[idx]
                confmat_metrics[str(idx)] = self.recall_precision_from_confmat(
                    conf_mat)

        return confmat_metrics

    @staticmethod
    def recall_precision_from_confmat(confmat: np.ndarray) -> dict:
        TP = confmat[1][1]
        TN = confmat[0][0]
        FP = confmat[0][1]
        FN = confmat[1][0]

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return {
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score
        }

    def get_all_preds(self, save_file: bool) -> list:
        for idx in range(len(self.results)):
            fnames_topk_ids = self.results[f'fold_{idx+1}']['fnames_topk_ids']
            pred_probs_lbls = []
            for filename, _ in fnames_topk_ids.items():
                fname_results = fnames_topk_ids[f'{filename}']
                original_fname = filename[:-16]

                pred = fname_results['pred'].detach().cpu().numpy()
                probs = fname_results['probs'].detach().cpu().numpy()
                lbl = fname_results['target'].detach().cpu().numpy()

                pred_probs_lbls.append((original_fname, probs, pred, lbl))

            df = pd.DataFrame(pred_probs_lbls)
            df.columns = ['original_fname', 'probs', 'pred', 'lbl']

            if save_file:
                df.to_csv(
                    f'{self.top_tiles_outdir}/fold-{idx+1}_probs_preds_lbls.csv',
                    sep='\t',
                    index=False
                )
            pred_probs_lbls.append(df)

            del df

        return pred_probs_lbls

    def get_metric_preds(self,
                         save_file: bool,
                         ) -> pd.DataFrame:

        pred_probs_lbls = []
        fnames_topk_ids = self.fold_metric_results['fnames_topk_ids']
        for filename, _ in fnames_topk_ids.items():
            fname_results = self.fold_metric_results[f'{filename}']
            original_fname = filename[:-16]

            pred = fname_results['pred'].detach().cpu().numpy()
            probs = fname_results['probs'].detach().cpu().numpy()
            lbl = fname_results['target'].detach().cpu().numpy()

            pred_probs_lbls.append((original_fname, probs, pred, lbl))

        df = pd.DataFrame(pred_probs_lbls)

        if save_file:
            df.to_csv(
                f'{self.top_tiles_outdir}/fold-{self.fold_num}_probs_preds_lbls.csv',
                sep='\t',
                index=False
            )

        return df

    def return_tiles_path(self,
                          features_filename: str) -> str:
        tiles_fname = features_filename[:-12]
        tiles_path = f'{self.tiles_dir}/{tiles_fname}.h5'
        return tiles_path

    def get_all_fold_results(
        self,
        metric: str,
        best_or_worst,
    ) -> int:
        """
        metrics:
            <recall>, <precision>, <f1_score>, <auprc>
        best_or_worst:
            best or worst fold, w/r/t  a metric
        Returns:
            Fold number with best/worst metric
        """
        recall = []
        precision = []
        f1_score = []
        auroc = []
        auprc = []
        conf_mats = []
        for idx in range(len(self.results)):
            fold_results = self.results[f'fold_{idx+1}']

            recall.append(
                fold_results['test_recall'].detach().cpu().numpy())
            precision.append(
                fold_results['test_precision'].detach().cpu().numpy())
            f1_score.append(
                fold_results['test_f1_score'].detach().cpu().numpy())
            auroc.append(
                fold_results['auroc'].detach().cpu().numpy()
            )
            auprc.append(
                fold_results['auprc'].detach().cpu().numpy())
            conf_mats.append(fold_results['confmat'].detach().cpu().numpy())

        final_results = {
            'final_recall': np.asarray(recall),
            'final_precision': np.asarray(precision),
            'final_f1_score': np.asarray(f1_score),
            'final_auroc': np.asarray(auroc),
            'final_auprc': np.asarray(auprc),
            'conf_mats': conf_mats
        }

        if best_or_worst == 'worst':
            fold_num = np.argmin(final_results[f'final_{metric}'])
        else:
            fold_num = np.argmax(final_results[f'final_{metric}'])

        final_results['fold_num'] = int(fold_num+1)

        return final_results

    @staticmethod
    def load_tiles(filepath: str) -> np.ndarray:
        with h5py.File(filepath, 'r') as f:
            tiles = np.asarray(f['tiles'], dtype=np.uint8)
        return tiles


    # Visualization settings
parser = argparse.ArgumentParser(
    description="Configurations for post-training analysis (e.g. visualization)"
)

parser.add_argument(
    '--tiles_dir',
    type=str,
    default=None,
    help='Directory for tessallated tiles'
)
parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='Directory for output'
)
parser.add_argument(
    '--model_results',
    type=str,
    default=None,
    help='Path for saved model results (e.g., evaluated metrics)'
)
parser.add_argument(
    '--fold_num',
    type=int,
    default=1,
    help='Retrieve results and tiles from fold number i (default: 1)'
)
parser.add_argument(
    '--get_all_preds',
    type=bool,
    default=False,
    help='Output all results from model to csv (default: False)'
)
parser.add_argument(
    '--get_top_ids',
    type=bool,
    default=False,
    help='Recover top tile ids without tile images (default: False)'
)
parser.add_argument(
    '--get_top_probs',
    type=bool,
    default=False,
    help='Recover top derived softmax probabilities (from Grad-CAM) (default: False)'
)
parser.add_argument(
    '--which_metric',
    type=str,
    default='f1_score',
    help='Retrieve results with respect to metric (<recall><precision><f1_score><auroc><auprc>) (default: f1_score)'
)

args = parser.parse_args()


def main(args):
    from joblib import Parallel, delayed

    viz = Visualization(
        tiles_dir=args.tiles_dir,
        top_tiles_outdir=args.output_dir,
        results_fp=args.model_results,
        metric=args.which_metric
    )

    if args.get_all_preds:
        viz.get_all_preds(save_file=True)

    if args.get_top_ids:
        return viz.get_top_tiles(save_tiles=True, id_only=True)

    if args.get_top_probs:
        return viz.get_attn_vals()

    # viz.get_top_tiles(
    #     save_tiles=True,
    #     fold_num=args.fold_num,
    #     id_only=False
    # )
    Parallel(n_jobs=10)(delayed(viz.get_top_tiles)(save_tiles=True,
                                                   fold_num=args.fold_num,
                                                   id_only=False))


if __name__ == '__main__':
    main(args)
