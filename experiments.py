import os
from itertools import product
from importlib import import_module
import configs

def get_experiment_dict(args, exp_name=None):
    # SEGMENTATION DA SEG
    expAll = {


    "wisenet_cp_iou":
        {
            "modelList": ["TwoHeads_iou"],
            "configList": ["basic_train"],
            "datasetList": ["CityScapesObject"],
            "metricList": ["AP50_segm"],
            "epochs": 1000
        },

    "wisenet_kitti_sharp":
        {
            "modelList": ["TwoHeads_sharpmask"],
            "configList": ["basic_train"],
            "datasetList": ["Kitti"],
            "metricList": ["AP50_segm"],
            "epochs": 1000
        },

    "wisenet_kitti_cosine":
        {
            "modelList": ["TwoHeads_cosine"],
            "configList": ["basic_train"],
            "datasetList": ["Kitti"],
            "metricList": ["AP50_segm"],
            "epochs": 1000
        },

    "wisenet_kitti":
        {
            "modelList": ["TwoHeads"],
            "configList": ["basic_train"],
            "datasetList": ["Kitti"],
            "metricList": ["AP50_segm"],
            "epochs": 1000
        }
    }

    if exp_name not in expAll:
        exp_dict = {
            "modelList": None,
            "configList": None,
            "datasetList": None,
            "metricList": None,
            "epochs": 1000
    }
    else:
        exp_dict = expAll[exp_name]
    # Override if needed
    exp_dict["configList"] = args.configList or exp_dict["configList"]
    exp_dict["metricList"] = args.metricList or exp_dict["metricList"]
    exp_dict["datasetList"] = args.datasetList or exp_dict["datasetList"]
    exp_dict["modelList"] = args.modelList or exp_dict["modelList"]
    
    return exp_dict


def get_main_dict_list(args, exp_name, mode):

    exp_dict = get_experiment_dict(args, exp_name)

    # Get Main Class
    project_name = os.path.realpath(__file__).split("/")[-2]
    MC = MainClass(
        path_models="models",
        path_datasets="datasets",
        path_metrics="metrics",
        path_samplers="addons/samplers.py",
        path_transforms="addons/transforms.py",
        path_saves="/mnt/projects/counting/Saves/main/iccv/",
        project=project_name)

    main_dict_list = []

    for model_name, config_name, metric_name, dataset_name in product(
            exp_dict["modelList"], exp_dict["configList"],
            exp_dict["metricList"], exp_dict["datasetList"]):


        config = configs.get_config_dict(config_name)

        # if key in key_set:
        #     continue

        # key_set.add(key)
        main_dict = MC.get_main_dict(
            mode, dataset_name, model_name, config_name, config, args.reset,
            exp_dict["epochs"], metric_name=metric_name)
        main_dict["config"] = config
        # main_dict["exp_dict"] = exp_dict

        main_dict_list += [main_dict]

    return main_dict_list


def get_module_classes(module_name):
    import inspect

    mod_dict = {}

    modList = import_module("{}.__init__".format(module_name)).__all__

    for module in modList:
        funcs = get_functions(module)
        for name in funcs:
            val = funcs[name]

            if not inspect.isclass(val):
                continue

            if (name in mod_dict and module_name in str(val.__module__)):
                if name != "Pascal2012":
                    raise ValueError("repeated %s" % name)
                print("Repeated:", name)
            mod_dict[name] = val

    return mod_dict


# Main dict
class MainClass:
    def __init__(self, path_datasets, path_models, path_samplers,
                 path_transforms, path_metrics, path_saves,
                 project):
        self.path_datasets = path_datasets
        self.path_saves = path_saves
        self.dataset_dict = get_module_classes(path_datasets)

        self.metric_dict = get_module_classes(path_metrics)
        self.model_dict = get_module_classes(path_models)

        self.sampler_dict = get_functions(path_samplers)
        self.transform_dict = get_functions(path_transforms)
        self.project = project


        # DATASETS

    def get_main_dict(self,
                      mode,
                      dataset_name,
                      model_name,
                      config_name,
                      config,
                      reset,
                      epochs,
                      metric_name,
                      gpu=None):

        main_dict = config
        # main_dict["exp_name"] = exp_name
        main_dict["config_name"] = config_name
        main_dict["model_name"] = model_name
        main_dict["metric_name"] = metric_name
        main_dict["dataset_name"] = dataset_name
        main_dict["epochs"] = epochs
        main_dict["reset"] = reset
        main_dict["project_name"] = self.project
        main_dict["code_path"] = "/mnt/home/issam/Research_Ground/{}".format(
            self.project)
        # GET GPU
        # set_gpu(gpu)
        main_dict["key"] = ("{} - {}".format(model_name, config_name),
                            "{}_({})".format(dataset_name, metric_name))
        main_dict["path_datasets"] = self.path_datasets
        main_dict["exp_id"] = (
            "dataset:{}_model:{}_metric:{}_config:{}".format(
                dataset_name, model_name, metric_name, config_name))

        # SAVE
        main_dict["path_save"] = "{}/{}/".format(self.path_saves,
                                                 main_dict["exp_id"])

        main_dict["path_summary"] = main_dict["path_save"].replace(
            "Saves", "Summaries")

        main_dict["metric_dict"] = self.metric_dict
        main_dict["sampler_dict"] = self.sampler_dict
        main_dict["model_dict"] = self.model_dict
        main_dict["dataset_dict"] = self.dataset_dict
        main_dict["transform_dict"] = self.transform_dict


        assert_exist(main_dict["model_name"], self.model_dict)
        assert_exist(main_dict["metric_name"], self.metric_dict)
        assert_exist(main_dict["dataset_name"], self.dataset_dict)

        return main_dict


def get_functions(module):
    import importlib
    if isinstance(module, str):
        spec = importlib.util.spec_from_file_location("module.name", module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    funcs = {}
    for name, val in module.__dict__.items():
        if name in funcs:
            raise ValueError("Repeated func %s" % name)

        if callable(val):
            funcs[name] = val

    return funcs


def assert_exist(key, dict):

    if key is None:
        return
    if key not in dict:
        raise ValueError("{} does not exist...".format(key))
