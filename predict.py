# Use a trained network to predict the flower name from an input image along with the probability of that name

'''
Basic usage: python predict.py /path/to/image checkpoint
Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu
Example usage: 
python predict.py flowers/test/1/image_06743.jpg assets
'''
# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models

# ------------------------------------------------------------------------------- #
# Function Definitions
# ------------------------------------------------------------------------------- #
# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Predict Image Classifier")

    # image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='image file for prediction.',
                        required=True)

    # checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='checkpoint file as str.',
                        required=True)
    
    # top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help=' categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda')

    # Parse args
    args = parser.parse_args()
    
    return args

# Function loads model from checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load("checkpoint.pth")
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_path):
    pil_img = PIL.Image.open(image_path)

    # Get original dimensions
    orig_width, orig_height = pil_img.size

    # Find shorter size and create settings to crop shortest side to 256
    if original_width < original_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    pil_img.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    pil_img = pil_img.crop((left, top, right, bottom))

    np_image = np.array(pil_img)/255 

    # Normalize each color channel
    Normalize_means = [0.485, 0.456, 0.406]
    Normalize_std = [0.229, 0.224, 0.225]
    np_image = (np_image-Normalize_means)/Normalize_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image


def predict(image_tensor, model, device, cat_to_name, top_k):
    if type(top_k) == type(None):
        top_k = 5
        print("Top K is not defined,default K=5.")

    model.eval();

    # Convert image from numpy to torch
    torchimage = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)

    model=model.cpu()

    log_probs = model.forward(torchimage)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    probs, labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    probs = np.array(probs.detach())[0]
    labels = np.array(labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    labels = [idx_to_class[label] for label in labels]
    flowers = [cat_to_name[label] for label in labels]
    
    return probs, labels, flowers


def print_prob(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    


# Main Function


def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    

    probs, labels, flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    

    print_prob(flowers, probs)

# Run Program

if __name__ == '__main__': main()