# FedBAGAN
**Notice**
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

This project is completed by capstone group CS47-2 and supervised by Ali Anaissi from The University of Sydney

## Methods

*Overall Federated Learning Structure*

<img src="https://user-images.githubusercontent.com/58716235/187176937-03e609bd-3b99-4432-a1c7-000a3fb16f9f.jpg " width="500" height="400" alt="Overall Federated Learning Structure"/><br/>

*BAGAN models charts in each client*
![detail](https://user-images.githubusercontent.com/58716235/187180779-584ae54f-b02f-4651-801b-1ad56dbc166a.jpg)

**Experimental flow chart**

<img src="https://user-images.githubusercontent.com/58716235/187187465-1db08548-30da-4d15-a3e9-10aed19e5ac5.jpg" width="500" height="600">

## Experimental Results

**Validation loss comparison**

<img src="https://user-images.githubusercontent.com/58716235/187187637-de6808eb-8c44-44c9-bba4-332373c5e083.jpg">

**FID evaluation**

The FID measures the similarity of two sets of images in terms of the statistical similarity of the computer vision features of the original image, with a lower score indicating that the two sets of images are more similar or that the statistics of the two are more similar. That is to say, lower FID score represent better performance in class imbalance problem.

Experimental setup for models | Overall FID score(for all 10 classes)
------------------------------|--------------------------------------
FEDGAN                        |   169.26
FEDBAGAN 150_SGD_NLL          |   96.42
FEDBAGAN 60_SGD_NLL           |   101.49
FEDBAGAN 100_ADAM_NLL         |   85.59
FEDBAGAN 100_ADAM_CEL         |   97.12
FEDBAGAN 100_SGD_CEL          |   97.13


## Reference
[1] *Cao, X., Sun, G., Yu, H., & Guizani, M. (2022). PerFED-GAN: Personalized Federated Learning via Generative Adversarial Networks. arXiv preprint arXiv:2202.09155.*

[2] *Mariani, G., Scheidegger, F., Istrate, R., Bekas, C., & Malossi, C. (2018). BAGAN: Data augmentation with balancing GAN. arXiv preprint arXiv:1803.09655.*

[3] *Rasouli, M., Sun, T., & Rajagopal, R. (2020). FEDGAN: Federated generative adversarial networks for distributed data. arXiv preprint arXiv:2006.07228.*

[4] *Wang, H., Kaplan, Z., Niu, D., & Li, B. (2020, July). Optimizing federated learning on non-iid data with reinforcement learning. In IEEE INFOCOM 2020-IEEE Conference on Computer Communications (pp. 1698-1707). IEEE.

[5] *Wang, L., Xu, S., Wang, X., & Zhu, Q. (2021, February). Addressing class imbalance in federated learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 11, pp. 10165-10173).

[6] *Zhang, W., Wang, X., Zhou, P., Wu, W., & Zhang, X. (2021). Client selection for federated learning with non-iid data in mobile edge computing. IEEE Access, 9, 24462-24474.
