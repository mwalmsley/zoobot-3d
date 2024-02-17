# Mask R-CNN for bars and spiral arms
Mask R-CNN using Zoobot as feature extraction backbone
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