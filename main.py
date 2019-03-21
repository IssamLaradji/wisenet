import glob, os
os.environ['TORCH_MODEL_ZOO'] = '/mnt/home/issam/pretrained/resnet'

import matplotlib
matplotlib.use('Agg')
# import multiprocessing 
# multiprocessing.set_start_method('spawn')
import cv2
cv2.setNumThreads(0) 

import torch
import argparse
import numpy as np
import experiments
import test
import subprocess
import misc as ms
import pandas as pd
import torch.backends.cudnn as cudnn
import borgy

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--expList', nargs="+", default=[None])
    parser.add_argument('-b', '--borgy', default=0, type=int)
    parser.add_argument('-br', '--borgy_running', default=0, type=int)
    parser.add_argument('-m', '--mode', default="summary")
    parser.add_argument('-r', '--reset', default=0, type=int)
    parser.add_argument('-s', '--status', type=int, default=0)
    parser.add_argument('-k', '--kill', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-c', '--configList', nargs="+", default=None)
    parser.add_argument('-l', '--lossList', nargs="+", default=None)
    parser.add_argument('-d', '--datasetList', nargs="+", default=None)
    parser.add_argument('-metric', '--metricList', nargs="+", default=None)
    parser.add_argument('-model', '--modelList', nargs="+", default=None)
    parser.add_argument('-p', '--predictList', nargs="+", default=None)
    parser.add_argument('-lr', '--load_result', default=0, type=int)
    parser.add_argument('-db', '--debug_mode', default=0, type=int)
    parser.add_argument('-logs', '--logs', default=0, type=int)
    parser.add_argument('-path', '--path', default=0, type=int)
    parser.add_argument('-num_workers', '--num_workers', default=0, type=int)
    parser.add_argument('-n_train', '--n_train', default=100, type=int)
    parser.add_argument('-n_val', '--n_val', default=-1, type=int)
    parser.add_argument('-mp', '--multiprocess', default=-1, type=int)
    parser.add_argument('-sm', '--summary_mode', default=0, type=int)
    parser.add_argument('-vm', '--visualize_mode', default=0, type=int)
    parser.add_argument('-retry', '--retry', default=0, type=int)
    args = parser.parse_args()

    # SEE IF CUDA IS AVAILABLE
    assert torch.cuda.is_available()
    print("CUDA: %s" % torch.version.cuda)
    print("Pytroch: %s" % torch.__version__)

    mode = args.mode
    results = {}
    if args.reset == 1 and args.borgy == 1:
        print("reset borgy job?")
        import ipdb; ipdb.set_trace()  # breakpoint 6737c7ec //
        

    for exp_name in args.expList:

        main_dict_list = experiments.get_main_dict_list(args, exp_name, mode)
        for main_dict in main_dict_list:
            main_dict["borgy_running"] = args.borgy_running
            main_dict["path_save"] = main_dict["path_save"] + mode + "/"
            main_dict["exp_name"] = exp_name
            
            # # SET SEED
            # np.random.seed(1)
            # torch.manual_seed(1)
            # torch.cuda.manual_seed_all(1)

            if args.path:
                print(main_dict["path_save"])
                continue

            if args.retry:
                pass

            if args.logs:
                results[main_dict["key"]] = borgy.borgy_logs(main_dict, mode)
                continue

            if args.status:

                results[(exp_name, "status")] = borgy.borgy_status(main_dict, mode)

                continue

            if args.kill:
                results[main_dict["key"]] = borgy.borgy_kill(main_dict, mode)
                continue

            if args.borgy:
                results[main_dict["key"]] = borgy.borgy_submit(
                    main_dict, mode, args.reset)
                continue
            if mode == "copy_best":
                import ipdb; ipdb.set_trace()  # breakpoint 2bb36824 //

                continue
            if mode == "backup":
                path_save = main_dict["path_save"]
                path_backup = "/mnt/projects/counting/Saves/main/backup/"
                copy_code = "cp -r %s "\
                            "%s" % (path_save, path_backup)

                subprocess.call([copy_code], shell=True)
                results[main_dict["key"]] = "copied to %s" % path_backup

            if mode == "sanity":
                from train_utils import sanity
                sanity.train(
                        main_dict,
                        reset=args.reset,
                        n_train=args.n_train,
                        n_val=args.n_val,
                        num_workers=args.num_workers,
                        multiprocess=args.multiprocess,
                        borgy_running=args.borgy_running,
                        debug_mode=args.debug_mode,
                        summary_mode=args.summary_mode)
                continue

            # if args.summary:
            #     row_key = ("{} - {} - {}".format(main_dict["model_name"],
            #                                      main_dict["config_name"],
            #                                      main_dict["loss_name"]))
            #     col_key = "{}-{}_({})".format(
            #         main_dict["config"]["dataset_src"],
            #         main_dict["config"]["dataset_tgt"],
            #         main_dict["metric_name"])
            #     main_dict["key"] = (row_key, col_key)

            #     main_dict["key"] = (row_key, "%s|iters"%main_dict["metric_name"])
            #     history = ms.load_model(
            #         main_dict, fname="train_latest", history_only=True)

            #     # metric_name = history["metric_name"]
            #     if "best_score" in history["summary"]:
            #         best_score = history["summary"]["best_score"]
            #     else:
            #         best_score = "-1"
            #     results[main_dict["key"]] = "{:s}|{:d}".format(
            #         best_score, history["iters"])

            #     # from train_utils import train_lifelong
            #     # ms.save_yaml("summary.yml", history["summary"])
            #     # results[main_dict["key"]] = "check summary.yml"
            #     # print("saved summary.yml")
            #     continue


            if mode == "train":
                from train_utils import train
                train.train(
                    main_dict,
                    reset=args.reset,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    num_workers=args.num_workers,
                    multiprocess=args.multiprocess,
                    borgy_running=args.borgy_running,
                    debug_mode=args.debug_mode,
                    summary_mode=args.summary_mode,
                    visualize_mode=args.visualize_mode)
     
                continue
            


            # if mode == "summary":
            #     history = ms.load_history(main_dict)

            #     result_dict = history["best_model"]
            #     results[main_dict["key"]] = "{:.2f}|{:d}-{:d}".format(
            #         result_dict[main_dict["metric_name"]], result_dict["epoch"], history["epoch"])
            #     # "{:.2f}-{:.2f}-{:.2f}".format(result_dict["0.25"],
            #     #   result_dict["0.5"], result_dict["0.75"])

    print(ms.dict2frame(results))


if __name__ == "__main__":
    main()