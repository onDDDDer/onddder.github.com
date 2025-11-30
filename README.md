DMFAA

This project implements DMFAA (Diversified Multi‑stage Flatness‑Aware Attack), a method to generate highly transferable adversarial examples for black-box image classification models.



📁 Project Structure
```text
├── main.py                # DMFAA main implementation
├── torch_nets/            # tf2torch converted models (Inception v3/v4)
├── data/
│   ├── images/            # image files
│   └── images.csv         # CSV file: [filename, label]
└── README.md
```

main.py reads data/images.csv and data/images/, generates adversarial samples in batches, and evaluates the attack success on a black-box model.


🔧 Requirements

- Python 3.8+
- PyTorch >= 1.7
- torchvision
- numpy, pandas, pillow


📦 Models

1. CNN-based Models (Converted from TensorFlow)
All models in this category are converted from TensorFlow to PyTorch using the MMdnn pipeline from:
[ylhz/tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model)

These include:
- Inception-v3 (Inc-v3)
- Inception-v4 (Inc-v4)
- Inception-ResNet-v2 (IncRes-v2)
- ResNet-50 (Res-50)
- ResNet-101 (Res-101)
- Inception-v3<sub>ens3</sub> (Inc-v3 ens3)
- Inception-v3<sub>ens4</sub> (Inc-v3 ens4)
- Inception-ResNet-v2<sub>ens</sub> (IncRes-v2 ens)
> These models include both standard ImageNet classifiers and ensemble adversarially trained variants widely used in transferability research.

2. Transformer-based & CNN Baseline Models (Hugging Face Model Hub)
All models in this category are downloaded from the Hugging Face model hub:
- [ViT-B (Vision Transformer Base)](https://huggingface.co/google/vit-base-patch16-224)
- [DeiT-B](https://huggingface.co/facebook/deit-base-patch16-224)
- [PiT-B (Pooling-based Vision Transformer)](https://huggingface.co/naver/pit-b-distilled-224)
- [Swin-T](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- [Swin-S](https://huggingface.co/microsoft/swin-small-patch4-window7-224)
- [LeViT](https://huggingface.co/facebook/levit-256)
- [CaiT-S](https://huggingface.co/facebook/cait-small-patch32-224)
- [ConViT-B](https://huggingface.co/facebook/convit-base)
- [Visformer-S](https://huggingface.co/naver/visformer-small)
- [DenseNet-121](https://huggingface.co/pytorch/vision-densenet-121)
- [VGG-19](https://huggingface.co/pytorch/vision-vgg19)


▶️ Running DMFAA

Run the attack with:
`python main.py`











