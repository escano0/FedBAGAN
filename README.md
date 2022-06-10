# FedBAGAN
A production of BAGAN model implimented on Federated Learning structure with pytorch.
The code can be run on colab by following instruction:
1. running_BAGAN.ipynb (Run code chunks from top to bottom, and skip the 'GAN training' chunk. 
ps:aotoencoder and Fed_BAGAN training would take lot of time)
2. inference.ipynb 
3. Calculate_FID.ipynb (To implimnet the fid calculatiion you should manually upload Cifar-10 jpg image set can divided them into 10 classes and paste the dirs to corresponding places)
4. visualize_loss.ipynb

Noticed that the saving dir and training dir in each code should be change due to actural saving dir

Modify BAGAN here

![da22664fe4e3cd09b37d3b2865d8476](https://user-images.githubusercontent.com/58716235/173056434-424a3967-1544-4a88-b340-14758a877b6a.png)
