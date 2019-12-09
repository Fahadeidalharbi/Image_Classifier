# Train a new network on a dataset and save the model as a checkpoint

'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
Example usage:
python train.py flowers --gpu --save_dir assets
'''


import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
#import workspace_utils 

def arg_parser():
# Define parser
    parser = argparse.ArgumentParser(description="Train Image Classifier ")
    

    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture')

    parser.add_argument('--save_dir', 
                      type = str, 
                      default = 'checkpoint.pth', 
                      help = 'Path to checkpoint')
    
#Set hyperparameters
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Set learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Set Hidden units as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

# Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
# Parse args
    args = parser.parse_args()
    return args

# Function transforming the train datasets
def data_transformer(train_dir):
   # Define transformation
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the Data
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    return train_data

# Function transforming the test/validation datasets
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
# Function for createing a dataloader from dataset imported
def train_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# model(architecture="vgg16") downloads model from torchvision
def define_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture is vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

# Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
def inputfeatures_hiddenlyer_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Hidden Layers is 4096.")
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Function validation for validating training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    test_accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        y_pred = model.forward(inputs)
        test_loss += criterion(y_pred, labels).item()
        
        ps = torch.exp(y_pred)
        equality = (labels.data == ps.max(dim=1)[1])
        test_accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, test_accuracy

# Function network_trainer represents the training of the network model
def Train_Network(model, trainloader, testloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 5
        print("Number of Epochs is 5.")    
 
    print("Training starting .....\n")

    #with active_session():
    for e in range(epochs):
        running_loss = 0
        model.train() # Technically not necessary, setting this for good measure

        for ii, (inputs, labels)  in enumerate(trainloader):
            steps = steps+ 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            y_pred = model.forward(inputs)
            train_loss = criterion(y_pred, labels)
            train_loss.backward()
            optimizer.step()

            running_loss = running_loss + train_loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, testloader, criterion, device)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(valid_accuracy/len(testloader)))

                running_loss = 0
                model.train()

    return model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def Test_Network(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Test Accuracy %d%%' % (100 * correct / total))

# Function for saves the model at a defined checkpoint
def init_checkpoint(Model, Save_Dir, Train_data):
       
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
          

            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, Save_Dir + '/checkpoint.pth')




# Function main() is where all the above functions are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_transformer(train_dir)
    valid_data = data_transformer(valid_dir)
    test_data = data_transformer(test_dir)
    
    trainloader = train_loader(train_data)
    validloader = train_loader(valid_data, train=False)
    testloader = train_loader(test_data, train=False)
    
    # Load Model
    model = define_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = inputfeatures_hiddenlyer_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learning rate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate is 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 5
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = Train_Network(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nDone!!")
    
    # Quickly Validate the model
    Test_Network(trained_model, testloader, device)
    
    # Save the model
    init_checkpoint(trained_model, args.save_dir, train_data)



# Run Program

if __name__ == '__main__': main()
