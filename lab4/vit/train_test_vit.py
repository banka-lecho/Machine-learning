from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.vit import VisionTransformer, LearnedPositionalEmbeddings, ClassificationHead, \
    PatchEmbeddings


class Configs(CIFAR10Configs):
    transformer: TransformerConfigs
    patch_size: int = 4
    n_hidden_classification: int = 2048
    n_classes: int = 10


@option(Configs.transformer)
def _transformer():
    return TransformerConfigs()


@option(Configs.model)
def _vit(c: Configs):
    d_model = c.transformer.d_model
    return VisionTransformer(c.transformer.encoder_layer, c.transformer.n_layers,
                             PatchEmbeddings(d_model, c.patch_size, 3),
                             LearnedPositionalEmbeddings(d_model),
                             ClassificationHead(d_model, c.n_hidden_classification, c.n_classes)).to(c.device)


def main():
    experiment.create(name='ViT', comment='cifar10')
    conf = Configs()
    experiment.configs(conf, {
        # Optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        # Transformer embedding size
        'transformer.d_model': 512,

        # Training epochs and batch size
        'epochs': 32,
        'train_batch_size': 64,

        # Augment CIFAR 10 images for training
        'train_dataset': 'cifar10_train_augmented',
        # Do not augment CIFAR 10 images for validation
        'valid_dataset': 'cifar10_valid_no_augment',
    })
    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
