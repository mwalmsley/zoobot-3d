# Mask R-CNN for bars and spiral arms
Mask R-CNN using Zoobot as feature extraction backbone.
* Folder `data` should contain two subfolders: `pngs` for the galaxy images (`{object_id}.png`) and `masks` for the pixel masks (`{object_id}_mask.png`) and a dataframe with object-ids, labels and bounding box coordinates for training.
* Folder `models` is the output folder for the logs and the model checkpoints.
Most things are setup via this config-file `Zoobot-backbone-transfer_config.json`:
```javascript
{
    // descriptive, not used
    "model_name": "Zoobot-backbone-transfer",
    "model_type": "Zoobot",
    // defines which ResNet-blocks of the backbone will be unfreezed for training
    // 0 - none, 1 - top/last block, 2 - top/last and 2nd last block, ..., 5 - all blocks
    "trainable_layers": 0,
    // descriptive, not used
    "description": "3-channel ResNet50 initialised with weights from Zoobot, transfer learning mode",
    // output directory for logs and checkpoints
    "log_dir": "./models/Zoobot-backbone-transfer/",
    // path to pre-trained checkpoint for the ResNet backbone, should be a Zoobot-checkpoint
    "pretrained_ckpt": "./pretrained_models/v2.0/gz-evo_300px_resnet50_rep1_div1_seed20622/checkpoints/epoch=23-step=92160.ckpt",
    // path to dataframe (parquet) with mask and label data
    "data_table": "./data/mask_labels.gzip",
    // path to directory containing the subdirectories /pngs and /masks
    "image_dir": "./data/",
    // batch size
    "batch_size": 16,
    // number of classes for the detector to classify. Should always includes +1 for background.
    "num_classes": 3,
    // number of epochs to traine
    "epochs": 150,
    // not used here
    "channels": 3,
    "bands": null,
    // mean pixel values for each channel
    "image_mean":
    [
        0.485,
        0.456,
        0.406
    ],
    // standard deviations of the pixel values for each channel
    "image_std":
    [
        0.229,
        0.224,
        0.225
    ],
    // not used here
    "k_folds": 0,
    "validation_split": 0.2,
    "test_split": 0.0,
    // name of the final model checkpoint in folder log_dir for validation and inference
    "final_model_ckpt": "MaskRCNN_Zoobot_epoch_104.pth"
}
```
## Main files
* `MaskRCNN_bars.ipynb` - Notebook illustrating the training of the Mask R-CNN detector
* `read_training_logs.ipynb` - Notebook for reading and plotting the output from the Tensorflow Summary writer
* `sample_prediction.ipynb` - Notebook showing a sample prediction of the model
* `mask_rcnn_dataset.py` - File containing the dataset-class for the Mask R-CNN model
## Data preparation
* `convert_data.ipynb` - Notebook for creating binary masks from the volunteers' markings
Mask R-CNN needs binary masks and an accompanying dataframe with object-ids, labels and bounding box coordinates for training. This notebook creates a set of training data from the available volunteers' data.
The aggregation process needs certainly much improvement.
## Helper files
* `coco_eval.py`
* `coco_utils.py`
* `engine.py`
* `transforms.py`
* `utils.py`
