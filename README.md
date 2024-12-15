# Multi-View-MLL Antenna Selection in mMIMO

We combined between two methods of MVL(Multi-View learning) ; a deep canonical
correlation DCCA and auto-encoders AE. in which the canonical relationship of the
compressed features and reconstruction errors of the auto-encoders are optimized
simultaneously.
Two loss functions are used: one is the loss measuring the correlation between
fm(hc) and ge(AS ), second is the loss at the output of the autoencoder. The total loss
function is : T= L1+alpha*L2
Network is consistent of three components: fx is feature mapping, fe the decoder
components (selected antennas is the input) treated as the second view.
The third component is the decoder designed to reconstruct the selected antenna .
The mathematical equation of both loss functions are found in the article.
Fx consists of 3 layers of 512 neurons
Both fd and fe have two layers of 512
Hyperparameters: batch size = 500
Alpha= 2
Lagrange = 0.5
Latent embeding dim= 51
The learning rate is decayed by 2% each epoch, and the value of momentum is given by
0.99.
