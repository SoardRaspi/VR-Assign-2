# # USAGE
# # python predict.py
# # import the necessary packages
# from imageSearch import config
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import cv2
# import os
# def prepare_plot(origImage, origMask, predMask):
# 	# initialize our figure
# 	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
# 	# plot the original image, its mask, and the predicted mask
# 	ax[0].imshow(origImage)
# 	ax[1].imshow(origMask)
# 	ax[2].imshow(predMask)
# 	# set the titles of the subplots
# 	ax[0].set_title("Image")
# 	ax[1].set_title("Original Mask")
# 	ax[2].set_title("Predicted Mask")
# 	# set the layout of the figure and display it
# 	figure.tight_layout()
# 	figure.show()

# def make_predictions(model, imagePath):
# 	# set model to evaluation mode
# 	model.eval()
# 	# turn off gradient tracking
# 	with torch.no_grad():
# 		# load the image from disk, swap its color channels, cast it
# 		# to float data type, and scale its pixel values
# 		image = cv2.imread(imagePath)
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		image = image.astype("float32") / 255.0
# 		# resize the image and make a copy of it for visualization
# 		image = cv2.resize(image, (128, 128))
# 		orig = image.copy()
# 		# find the filename and generate the path to ground truth
# 		# mask
# 		filename = imagePath.split(os.path.sep)[-1]
# 		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
# 			filename)
# 		# load the ground-truth segmentation mask in grayscale mode
# 		# and resize it
# 		gtMask = cv2.imread(groundTruthPath, 0)
# 		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
# 			config.INPUT_IMAGE_HEIGHT))


# # make the channel axis to be the leading one, add a batch
# 		# dimension, create a PyTorch tensor, and flash it to the
# 		# current device
# 		image = np.transpose(image, (2, 0, 1))
# 		image = np.expand_dims(image, 0)
# 		image = torch.from_numpy(image).to(config.DEVICE)
# 		# make the prediction, pass the results through the sigmoid
# 		# function, and convert the result to a NumPy array
# 		predMask = model(image).squeeze()
# 		predMask = torch.sigmoid(predMask)
# 		predMask = predMask.cpu().numpy()
# 		# filter out the weak predictions and convert them to integers
# 		predMask = (predMask > config.THRESHOLD) * 255
# 		predMask = predMask.astype(np.uint8)
# 		# prepare a plot for visualization
# 		prepare_plot(orig, gtMask, predMask)


# # load the image paths in our testing file and randomly select 10
# # image paths
# print("[INFO] loading up test image paths...")
# imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
# imagePaths = np.random.choice(imagePaths, size=10)
# # load our model from disk and flash it to the current device
# print("[INFO] load up model...")
# unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)
# # iterate over the randomly selected test image paths
# for path in imagePaths:
# 	# make predictions and visualize the results
# 	make_predictions(unet, path)


##################################


# USAGE
# python predict.py
# import the necessary packages
from imageSearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

# Calculate IoU (Intersection over Union)
def calculate_iou(pred_mask, gt_mask):
    # Convert masks to binary format
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

# Calculate Dice coefficient
def calculate_dice(pred_mask, gt_mask):
    # Convert masks to binary format
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Calculate intersection and sum of areas
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    sum_areas = pred_mask.sum() + gt_mask.sum()
    
    # Calculate Dice
    dice = (2 * intersection) / sum_areas if sum_areas > 0 else 0
    return dice

def prepare_plot(origImage, origMask, predMask, iou, dice, imagePath=None):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask, cmap='gray')
    ax[2].imshow(predMask, cmap='gray')
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title(f"Predicted Mask\nIoU: {iou:.4f}, Dice: {dice:.4f}")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.show()
    
    # Save the figure if requested
    if imagePath:
        os.makedirs("visualizations", exist_ok=True)
        save_path = os.path.join("visualizations", f"segmentation_{os.path.basename(imagePath).split('.')[0]}.png")
        figure.savefig(save_path)
        print(f"Visualization saved to {save_path}")

def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        print(f"Processing image: {imagePath}")
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
            filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_HEIGHT))
        
        # make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)
        
        # Calculate IoU and Dice scores
        iou = calculate_iou(predMask, gtMask)
        dice = calculate_dice(predMask, gtMask)
        print(f"File: {filename}, IoU: {iou:.4f}, Dice: {dice:.4f}")
        
        # prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask, iou, dice, imagePath)
        
        return iou, dice

# Create a directory for output visualizations
os.makedirs("visualizations", exist_ok=True)

# Initialize lists to store metrics
all_ious = []
all_dice_scores = []

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
print(f"Number of test images: {len(imagePaths)}")
imagePaths = np.random.choice(imagePaths, size=10)
print(f"Selected 10 random images for evaluation")

# load our model from disk and flash it to the current device
print("[INFO] loading model...")
try:
    unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# iterate over the randomly selected test image paths
print("Starting prediction on test images...")
for path in imagePaths:
    try:
        iou, dice = make_predictions(unet, path)
        all_ious.append(iou)
        all_dice_scores.append(dice)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Calculate and print average metrics
avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0
avg_dice = sum(all_dice_scores) / len(all_dice_scores) if all_dice_scores else 0
print(f"\nAverage metrics across {len(all_ious)} images:")
print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice score: {avg_dice:.4f}")