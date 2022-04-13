import sys
sys.path.append('./models')
from deeplab_3d import Deeplab
from prepare_tools import ChannDataset
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import numpy as np
from datetime import datetime

def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def imgReshape(img):
    newNum = img.shape[0]*img.shape[1]
    Nchann = img.shape[2]
    Ndim = img.shape[3]
    return img.reshape((newNum,Nchann,Ndim,Ndim,Ndim))

def labelReshape(label):
    newNum = label.shape[0]*label.shape[1]
    Ndim = label.shape[2]
    return label.reshape((newNum,Ndim,Ndim,Ndim))

def train(model, criterion, optimizer, epochs):
    isSch = False
    if isSch:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                             milestones=[epochs//2, epochs//4*3], gamma=0.1)
    writer1 = SummaryWriter(log_dir='log/train',comment='Train_loss')
    writer2 = SummaryWriter(log_dir='log/valid',comment='Valid_loss')
    #writer = SummaryWriter(log_dir='log/')
    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        total_train_loss = []
        model.train()
        for i, (img,labels) in enumerate(train_loader,0):
            if (i+1)%10==0:
                print("Epoch %d, batch %d, train"%(epoch+1,i+1))
            img = imgReshape(img)
            labels = labelReshape(labels)
            inputs = img.type(torch.FloatTensor)
            labels = labels.long()

            if CUDA:
                inputs,labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())
        train_loss.append(np.mean(total_train_loss))

        total_valid_loss = []
        model.eval()
        for i, (img,labels) in enumerate(valid_loader,0):
            if (i+1)%10==0:
                print("Epoch %d, batch %d, valid"%(epoch+1,i+1))
            img = imgReshape(img)
            labels = labelReshape(labels)
            inputs = img.type(torch.FloatTensor)
            labels = labels.long()
            if CUDA:
                inputs,labels = inputs.cuda(),labels.cuda()
            #with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}').format((epoch + 1), epochs,train_loss[-1],valid_loss[-1])

        print(log_string)
        log('./DeepLab.log',log_string)

        if isSch:
            scheduler.step()

        #writer.add_scalars('log',{'train_loss':train_loss[-1],'valid_loss':valid_loss[-1]},epoch)

        writer1.add_scalar('train_loss',train_loss[-1],epoch)
        writer2.add_scalar('valid_loss',valid_loss[-1],epoch)
        
        if (epoch+1)%10==0:
            file_path = './check/model%03d.pth'%(epoch+1)
            state = {'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
            torch.save(state,file_path)
    writer1.close()
    writer2.close()


if __name__ == '__main__':
    n1, n2, n3 = 128, 128, 128
    params = {'batch_size': 1,
              'dim': (n1, n2, n3),
              'n_channels': 1,
              'shuffle': True}

    tpathname = '/data/train/'
    vpathname = '/data/valid/'

    tdpath = tpathname+'sx/'
    tfpath = tpathname+'chann/'

    vdpath = vpathname+'sx/'
    vfpath = vpathname+'chann/'

    print("Start loading")

    bs = 4

    training_generator = ChannDataset(dpth=tdpath, fpth=tfpath, dimension=(n1,n2,n3), chann=1,num_sample=100)
    validate_generator = ChannDataset(dpth=vdpath, fpth=vfpath, dimension=(n1,n2,n3), chann=1,num_sample=20)
    # training_generator = testDataset(dpath=tdpath, fpath=tfpath,data_IDs=tdata_IDs, **params)
    #validation_generator = dataset(dpath=vdpath,fpath=vfpath,data_IDs=vdata_IDs,**params)

    train_loader = DataLoader(dataset=training_generator, batch_size=bs, shuffle=False,drop_last=True)
    valid_loader = DataLoader(dataset=validate_generator, batch_size=bs, shuffle=False,drop_last=True)

    print("Finish loading")

    is_loaded = False
    loaded_file = './check0505test/model150.pth'
    
    model = Deeplab(num_classes=2,output_stride=16,sync_bn=False,freeze_bn=False)
    if is_loaded :
        model.load_state_dict(torch.load(loaded_file)['net'])
        print("Load the pretrained model.")
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    else:
        model = model
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if is_loaded :
        optimizer.load_state_dict(torch.load(loaded_file)['optimizer'])
    train(model,criterion,optimizer,epochs=200)
