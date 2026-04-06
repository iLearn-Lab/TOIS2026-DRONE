import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import yaml
import torch
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config, n_params, get_now
from src.utils.vl_utils import GloVe
from src.experiment.eval import Evaluator
from src.model.model import Model


def train(config):
    # save log

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    exp_folder_path = os.path.join(config["exp_dir"], get_now())
    Path(exp_folder_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_folder_path, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # prepare data
    dataset_name = config["dataset_name"]
    data = prepare_data(config, dataset_name)
    train_dl = data["train_dl"]
    valid_dl = data["valid_dl"]
    test_dl = data["test_dl"]


    model = Model(config)
    # model = Model(config, vocab, glove)
    print("Model has {} parameters.\n".format(n_params(model)))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    test_evaluator = Evaluator()
    log_file_path = os.path.join(exp_folder_path, "train.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("{}\n".format(config[dataset_name]["feature_dir"]))
        log_file.write("Model has {} parameters.\n".format(n_params(model)))
        log_file.flush()
        for epoch in range(1, config[dataset_name]["epoch"] + 1):
            for i, batch in tqdm(
                enumerate(train_dl), total=len(train_dl),
                desc="Training epoch {} with lr {}".format(epoch, model.optimizer.param_groups[0]["lr"])
            ):
                model.train_mode()
                loss = model.forward_train_val(batch,epoch, config[dataset_name]["epoch"] + 1)
                model.optimizer_step(loss)

                # if use_gpu:
                #     model.batch_to(batch, model.cpu_device)
                #     torch.cuda.empty_cache()

            with torch.no_grad():
                test_loss = test_evaluator.eval_dataloader(model, test_dl,epoch,config[dataset_name]["epoch"] + 1)
                model.scheduler_step(test_loss)

            log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## test ##\n")
            log_file.write(test_evaluator.report_current() + "\n")
            log_file.write(test_evaluator.report_best() + "\n")
            log_file.flush()

            # save best
            if epoch == test_evaluator.best_epoch:
                model.save_checkpoint(exp_folder_path, "best")
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--task", help="Name of the dataset: {activitynetcaptions, charadessta, tacos}.", required=True)
    parser.add_argument('--gpu', type=str, default="1", help='GPU device id(s) to use, e.g. "0" or "0,1"')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--alpha', type=int, default=4, help='Weight multiplier for generation loss (gen_loss)')
    parser.add_argument('--beta', type=int, default=3, help='Weight multiplier for affinity alignment loss (affine_loss)')
    parser.add_argument('--gamma', type=float, default=0.25, help='Weight multiplier for frame-level contrastive loss (frame_loss)')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config("src/config.yaml")
    config["dataset_name"] = args.task
    config["gpu"] = args.gpu
    config["seed"] = args.seed
    config["alpha"] = args.alpha
    config["beta"] = args.beta
    config["gamma"] = args.gamma
    train(config)
