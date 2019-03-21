

def get_config_dict(config_name):
    configAll = {
    "basic_train":
             {"val_batchsize" :1,
            "dataset_src":"synthiaDataSet",
            "dataset_tgt":"cityscapes13",
            "loss_da_name":"lsd_loss",
            "batch_size": 1,
            "epochs": 500,

            "train_mode":"basic",

            "sampler_name": "Random1000",
            "epoch2val":5,
            "trainTransformer":"rgbNormalize",
            "testTransformer":"rgbNormalize",
            
            "model_options":{},
            "dataset_options":{"mold_image":1},
            "verbose":True},
            

    "debug": {
        "val_batchsize": 1,
        "opt_name": "adam",
        "batch_size": 1,
        "epochs": 500,
        "sampler_name": "Random10",
        "epoch2val": 5,
        "trainTransformer": "Tr_WTP_NoFlip",
        "testTransformer": "Te_WTP",
        "opt_options": {
            "lr": 1e-5,
            "weight_decay": 0.0005
        },
        "model_options": {},
        "dataset_options": {},
        "verbose": True
    }}
    

    return configAll[config_name]