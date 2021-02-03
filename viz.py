import argparse
import cifar_classes
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import models.cifar as models
import torch, os
import torch.nn as nn
import torch.nn.parallel
from collections import defaultdict
  
def extract_label(data, class_label,model):
    labels = []
    for batch_idx, (inputs, targets) in enumerate(data):
        for idx, image in enumerate(inputs):
            if  targets[idx].item() == class_label:
                label = model(image.unsqueeze(0))
                labels.append(label)
    return labels

    
def classify_class(data,class_labels,threshold):
    # return -1 for "new class"
    pred = defaultdict(int)
    for label in data:
        min_dist = float("inf")
        p = -1
        for idx, target in enumerate(class_labels):
            dist = torch.sum(torch.abs(target-label))
            if dist < min_dist and dist < threshold:
                min_dist = dist
                p = idx
        pred[p]+=1
    return pred    
    
    


parser = argparse.ArgumentParser(description='Continual Learning Visualization Tool')

parser.add_argument('--new-class', type=str, default='train',
                    help='An unseen class from CIFAR-100')

parser.add_argument('--model', default=[], type=str, nargs='+', metavar='PATH',
help='path to a model trained on CIFAR-10')

parser.add_argument('--similar-to', type=str, nargs='+', default=["truck"],
                        help='list of similar classes in CIFAR-10')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help= 'model architechture'  )                      
parser.add_argument('--depth', type=int, default=32, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--dense-labels', default=True, action='store_true',help='Use dense labels.')
                    
                    
args = parser.parse_args()

# convert new class to its proper index
if args.new_class not in cifar_classes.coarse_label:
    raise ValueError("Error: --new-class argument is not in CIFAR-100")
new_class = cifar_classes.coarse_label.index(args.new_class)

if len(args.similar_to) == 0:
    raise ValueError("Error: --similar-too argument must be given a value")

# convert similar classes to their proper indices
similar_classes = []
for class_name in args.similar_to:
    if class_name not in cifar_classes.cifar_10_labels:
        raise ValueError("Error: --similar-too list contains classes not found in CIFAR-10")
    similar_classes.append(cifar_classes.cifar_10_labels.index(class_name))

# load cifar-100
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataloader = datasets.CIFAR100
num_classes = 100
trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)    
def evaluate_model(model_name):




    # create model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                    categorical= not args.dense_labels
                )         
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()


    # Load model
    print("loading model into memory")
    assert os.path.isfile(model_name), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model_name)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if 'labels' in checkpoint and checkpoint['labels'] is not None:
        model.module.labels.set_labels(checkpoint['labels'])

    print("running model on new class")
    # get model outputs for the new class
    d = extract_label(testloader,new_class,model) # + extract_label(trainloader,new_class,model)

    # get results for a range of thresholds
    thresholds = []
    for i in range(60000,120000,2500):
        thresholds.append(i)
    results = dict()
    print("getting results for thresholds:")
    for threshold in thresholds:
        print(threshold)
        results[threshold] = classify_class(d,model.module.labels.dense_labels,threshold)


    # calculate metrics
    print("calculating metrics")
    metrics = dict()

    num_unknowns = []
    num_accurates = []
    num_inaccurates = []

    for threshold in thresholds:
        result = results[threshold]
        num_accurate = 0
        for similar_class in similar_classes:
            num_accurate += result[similar_class]
      
        num_unknown = result[-1]

        total = 0
        for key,val in result.items():
            total += val
            
        num_inaccurate = total - num_accurate - num_unknown
        
        metrics[threshold] = (num_unknown,num_accurate,num_inaccurate,total)
        num_accurates.append(num_accurate)
        num_unknowns.append(num_unknown)
        num_inaccurates.append(num_inaccurate)
    return thresholds, num_accurates, num_unknowns, num_inaccurates
# plot

for model_name in args.model:
    thresholds, num_accurates, num_unknowns, num_inaccurates = evaluate_model(model_name)
    plt.title(model_name)
    plt.plot(thresholds, num_accurates, 'g', label="num_accurates")
    plt.plot(thresholds, num_unknowns, 'b', label="num_unknowns")
    plt.plot(thresholds, num_inaccurates, 'r', label="num_inaccurates")
    plt.plot(thresholds, num_accurates, 'go')
    plt.plot(thresholds, num_unknowns, 'bo')
    plt.plot(thresholds, num_inaccurates, 'ro')   
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
    plt.savefig(model_name + "-" +args.new_class + '.png')
    
    
    
  
    
    
    
    
    
    
    
    