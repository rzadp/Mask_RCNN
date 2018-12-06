"""
Mask R-CNN
Badminton
"""

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BadmintonConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "badminton"

    # LEARNING_RATE = 0.0001
    # LEARNING_MOMENTUM = 0.09

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + badminton

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


    # MASK_SHAPE = [28, 28]
    # MASK_SHAPE = [56, 56]
    # MASK_SHAPE = [112, 112]
    MASK_SHAPE = [224, 224]

    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (112, 112)

    RPN_TRAIN_ANCHORS_PER_IMAGE = 15
    TRAIN_ROIS_PER_IMAGE = 10
    IMAGE_MAX_DIM = 512
    MAX_GT_INSTANCES = 5
    POST_NMS_ROIS_INFERENCE = 100
    POST_NMS_ROIS_TRAINING = 100

    BACKBONE = "resnet50"
    DETECTION_MAX_INSTANCES = 5



############################################################
#  Dataset
############################################################

class BadmintonDataset(utils.Dataset):

    def load_badminton(self, dataset_dir, subset):
        """Load a subset of the Badminton dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("badminton", 1, "badminton")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = annotations["_via_img_metadata"]
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "badminton",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a badminton dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "badminton":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "badminton":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epochs, use_multiprocessing):
    """Train the model."""
    # Training dataset.
    dataset_train = BadmintonDataset()
    dataset_train.load_badminton(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BadmintonDataset()
    dataset_val.load_badminton(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                use_multiprocessing=use_multiprocessing,
                layers='heads')
    model_path = os.path.join(ROOT_DIR, "samples/badminton/mask_rcnn_badminton.h5")
    print("Saving weights...")
    model.keras_model.save_weights(model_path)
    print("Saved!")


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect badminton.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/badminton/dataset/",
                        help='Directory of the Badminton dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--epochs', required=True,
                        metavar="20",
                        help="Number of epochs")
    parser.add_argument('--steps_per_epoch', required=False,
                        default=1000,
                        metavar="1000",
                        help="Steps per epoch")
    parser.add_argument('--validation_steps', required=False,
                        default=50,
                        metavar="50",
                        help="Validation steps")
    parser.add_argument('--use_multiprocessing', required=True,
                        metavar="False",
                        help="Should use multiprocessing")
    parser.add_argument('--verbose', required=False,
                        default=0,
                        metavar="0",
                        help='Verbosity')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    assert args.command == "train"
    assert args.dataset, "Argument --dataset is required for training"

    print("Logs: ", args.logs)

    # Configurations
    config = BadmintonConfig()
    config.STEPS_PER_EPOCH = int(args.steps_per_epoch)
    config.VALIDATION_STEPS = int(args.validation_steps)
    # config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    start = time.time()
    train(model, int(args.epochs), args.use_multiprocessing == "True")
    end = time.time()
    print("Done. train(model) took " + str(round((end - start) / 60, 1)) + " minutes")
