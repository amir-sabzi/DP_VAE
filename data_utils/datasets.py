import configlib


from torchvision import transforms
from data_utils.dataloader import NumpyLoader, Normalize, Cast, AddChannelDim
from torchvision.datasets import MNIST
from utils.logging_utils import ensure_dir

def get_dataset(config: configlib.Config):
    if config.dataset == "MNIST":
        return get_mnist(config)
    else:
        raise NotImplementedError("The getter for dataset is not implemented!")


def get_mnist(c: configlib.Config):
    normalize = [
        #  transforms.RandomCrop(28, padding=4),
        Cast(),
        Normalize([0.5, ], [0.5, ]),
        AddChannelDim(),
    ]
    transform = transforms.Compose(normalize)

        # Load MNIST dataset and use Jax compatible dataloader
    ensure_dir(c.data_dir)
    mnist = MNIST(c.data_dir, download=True, transform=transform)
    im_dataloader = NumpyLoader(
        mnist,
        batch_size=c.vae_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    mnist_test = MNIST(c.data_dir, train=False, download=True, transform=transform)
    im_dataloader_test = NumpyLoader(
        mnist_test,
        batch_size=c.vae_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    return mnist, mnist_test, im_dataloader, im_dataloader_test
    