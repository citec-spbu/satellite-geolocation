# пример кофиига для использования модели
# то что нужно менять - написанно на русском
model = dict(
    backbone=dict(
        type="CvT",                  # Backbone architecture used in the model.
        vit_type="cvt13",                  # Specific variant for MixFormer.
        pretrain_path="checkpoints/CvT-13-384x384-IN-22k.pth",  # путь к весам предобученной модели для инициализации бэкбона
        pretrain=True,                       # Use pretrained weights for backbone initialization.
        output_index=[0, 1, 2],              # Which backbone feature maps to use.
    ),
    neck=dict(
        type="CCN",
        output_dims=128,
        UAV_output_index=[0],
        Satellite_ouput_index=0,
    ),
    head=dict(
        type="ChannelEmbedding",           # Head type that computes the response map.
        input_ndim=128,                      # Input feature channels for the head.
        mid_process_channels=[64, 16, 1],    # Internal head channel sizes.
    ),
    postprocess=dict(
        upsample_to_original=True,           # Upsample outputs to final image size.
        upsample_method="NearstUpsample",   # Upsampling method.
        output_size=[576, 576],              # Final satellite output size for visualization.
    ),
    loss=dict(
        cls_loss=dict(
            type="BalanceLoss",            # Classification loss used for training.
            center_R=31,
            neg_weight=130,
        ),
    ),
)

# Data settings only matter for image sizes and normalization.
# In the inference wrapper we use UAVhw and Satellitehw to preprocess inputs.
data_config = dict(
    batchsize=8,
    num_worker=8,
    val_batchsize=8,
    train_dir="/path/to/train",            # для обучения,не нужно для инференса.
    val_dir="/path/to/val",                # для валидации,не нужно для инференса.
    test_dir="/path/to/test",              # Путь к папке с тестовыми данными. Структура папки должна быть такой же, как в обучении, с подкаталогами 'query_satellite' и 'query_drone'.
    test_mode="merge_test_700-1800_cr0.95_stride100",
    UAVhw=[192, 192],                        # Resize UAV input images to this size.
    Satellitehw=[576, 576],                  # Resize satellite input images to this size.
)

# Pipeline config contains augmentation definitions for training only.
pipline_config = dict(
    train_pipeline=dict(
        UAV=dict(
            RandomErasing=dict(probability=0.3),
            RandomResize=dict(img_size=data_config["UAVhw"]),
            ToTensor=dict(),
        ),
        Satellite=dict(
            RandomCrop=dict(cover_rate=0.85, map_size=(512, 1000)),
            RandomResize=dict(img_size=data_config["Satellitehw"]),
            ToTensor=dict(),
        ),
    ),
)

lr_config = dict(
    lr=1.5e-4,                               # Learning rate for training.
    type="cosine",                         # LR scheduler type.
    warmup_iters=500,
    warmup_ratio=0.01,
)

train_config = dict(autocast=True, num_epochs=12)

test_config = dict(
    num_worker=8,
    filterR=51,                              # Optional post-filter radius for response maps.
    checkpoint="output/net_best.pth",      # путь к весам модели для инференса.
)

checkpoint_config = dict(
    interval=1,
    epoch_start_save=6,
    only_save_best=True,
)

log_interval = 50
load_from = None
resume_from = None
debug = True
seed = 666
