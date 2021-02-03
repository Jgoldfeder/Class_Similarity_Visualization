''' Utils for dense label classification
Author: Judah Goldfeder
'''
import torch
import torch.nn as nn

# takes input of shape (batch,64) and outputs one of shape (batch,64,64)        
def get_decoder():

    deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
    return deconvs


class Labels:
    
    # pass in the number of classes, and label dimensions
    def __init__(self, num_classes,label_dim):
        self.num_classes = num_classes
        self.label_dim = label_dim
        self.dense_labels = []
        # initialize labels with Random Noise
        # uniform distribution in range [0,80)
        for i in range(num_classes):
            label = torch.rand(self.label_dim)*80
            self.dense_labels.append(label)
        
        self.dense_labels = torch.stack(self.dense_labels)
        
        # create embedding layer 
        self.embedding = nn.Embedding.from_pretrained(self.dense_labels,freeze=True)

    # move the model to cuda
    def cuda(self):
        if torch.cuda.is_available():
            self.dense_labels = self.dense_labels.cuda()
            self.embedding = self.embedding.cuda()
            
    # given a tensor of labels represented as 0,1,2,3... etc, give back a new tensor with the dense representation    
    def get_dense(self,labels):
        return self.embedding(labels)
    
    def set_labels(self,labels):
        self.dense_labels = labels
        self.embedding = nn.Embedding.from_pretrained(self.dense_labels,freeze=True)

    # given dense labels, get the nearest categorical matches
    def get_categorical(self,labels): 
        batch_size = labels.shape[0]
     
        # shape of self.dense_labels is (num_classes , label_dim)
        # lets make it (batch_size , label_dim , num_classes)
        x = self.dense_labels.permute(1,0).unsqueeze(0).expand(batch_size,self.label_dim,self.num_classes)
        
        # shape of labels is (batch_size , label_dim)
        # lets make it (batch_size , label_dim , num_classes)
        y = labels.unsqueeze(2).expand(batch_size,self.label_dim,self.num_classes)
        
        # now subtract and get abs
        c = torch.abs(x-y)
        
        # finally, sum along dimension 1
        c = torch.sum(c,1)
        
        # c is now of shape (batch_size , num_classes) :)
        # negate c so that higher number means more likely, which is the convention
        return -c
    
    def update(self,new_labels,lr):
        # move towards label average
        self.dense_labels = self.dense_labels.cuda() + lr*(new_labels.cuda() - self.dense_labels.cuda())
        # move away from other labels
       
        other_label_avg = (torch.sum(new_labels,0,True).expand(self.num_classes,self.label_dim) - new_labels)/(self.num_classes-1)
        
        
        
        self.dense_labels = self.dense_labels.cuda() - lr*(other_label_avg.cuda() - self.dense_labels.cuda())
        self.embedding = nn.Embedding.from_pretrained(self.dense_labels,freeze=True)
     
    def save(self,name):
        torch.save(self.dense_labels,name)