

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from imgaug import augmenters as iaa

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGENET50_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_2.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class ShipConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ship"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + ship

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
    IMAGE_MIN_DIM = 200
    IMAGE_MAX_DIM = 256
    
    #BACKBONE = "resnet50"
    
    VALIDATION_STEPS = 1
    
    #IMAGE_RESIZE_MODE = "none"

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class ShipDataset(utils.Dataset):

    def load_ship(self, dataset_dir, subset):
        """Load a subset of the ship dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("ship", 1, "ship")

        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)


        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys


        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                name = [r['region_attributes']['type'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                name = [r['region_attributes']['type'] for r in a['regions']] 
                
            #Class
            name_dict = {"ship":1}
            name_id = [name_dict[a] for a in name]
            

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "ship",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=name_id,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)
        
        name_id = image_info["class_id"]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1


        return (mask.astype(np.bool), class_ids)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ship":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ShipDataset()
    dataset_train.load_ship(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShipDataset()
    dataset_val.load_ship(args.dataset, "val")
    dataset_val.prepare()

    
    max_augs = 3
    augmentations_ship = iaa.SomeOf((0, max_augs), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([ iaa.Affine(rotate = 30 * i) for i in range(0, 12) ]),
    iaa.Add((-40, 40)),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='all,
                augmentation=augmentations_ship)




############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ships.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/ship/dataset/",
                        help='Directory of the ship dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ShipConfig()
    else:
        class InferenceConfig(ShipConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
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
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
