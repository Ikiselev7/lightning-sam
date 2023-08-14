from box import Box

config = {
    "num_devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/home/ilia_kiselev/Downloads/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": False,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
        "chunk_size": 16,
    },
    "dataset": "/home/ilia_kiselev/hubmap_organ_yolo_folds_3_cls/dataset.yaml",
    "yolo_cfg": "/home/ilia_kiselev/lightning-sam/lightning_sam/yolo.yaml"
}

cfg = Box(config)
