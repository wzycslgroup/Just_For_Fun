# Some tricks about GAN

- Learning rate should be set to about 1e-4
- we should first fix generator and train discriminator for k times, then we fix discriminator and train generator for only few times which should be less than k
- we need to scale up the input image to [-1, 1] before training
- leakyrelu
- batch_size
- latent_dim