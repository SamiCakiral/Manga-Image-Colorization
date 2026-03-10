# 🎨 Manga Colorization with Deep Learning

A Google Colab project for automatically colorizing black and white manga pages using various deep learning approaches.

## 🌟 About

This project aims to colorize black and white manga pages using different neural network architectures, all packaged in an easy-to-use Google Colab notebook. It's designed to be accessible to both beginners and advanced users through form-based configuration.

## 🚀 Quick Start

1. Open the notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ICz5vqmkn38vLCC7hhkaVSQNkDVGrxYf#scrollTo=S5J9Ea6HB4v9)

2. Follow the notebook sections in order:
   - Setup & Dataset
   - Model Configuration
   - Training
   - Inference & Results

## 📊 Available Models

All models can be selected and configured through simple form interfaces:

- **UNet**: Basic but effective architecture
- **ResUNet**: Enhanced version with residual connections
- **VAE**: For style-consistent colorization
- **GAN**: For more realistic results
- **Diffusion**: Latest approach for high-quality results

## ⚙️ Configuration

### Dataset Configuration
```
• Batch size: Slider (1-16)
• Epochs: Slider (5-100)
• Learning rate: Number input
• Target images: Number input
• Target size: Slider (256-1024)
• Skip pages: Start/End page settings
```

### Model Selection
```
• Model type: Dropdown menu
• Architecture-specific parameters:
  - Number of filters
  - Use of attention
  - Dropout rates
  - etc.
```

### Training Options
```
• Load pretrained: Checkbox
• Continue training: Checkbox
• Save frequency: Slider
• Show samples: Checkbox
• Plot loss: Checkbox
```

## 💾 Dataset

The project includes:
- Automatic dataset download functionality
- Built-in preprocessing pipeline
- Support for CBZ manga files
- Automatic train/validation split

## 🖼️ Results

Results are automatically organized in:
- Grayscale images
- Colorized outputs
- Original color images
- Side-by-side comparisons

## 🚨 Requirements

- A Google account
- Access to Google Colab
- GPU runtime enabled (Settings > Hardware accelerator > GPU) (if you want to train a model, otherwise the CPU can run inferences)

## ⚠️ Important Notes

- Save your trained models to Google Drive
- The free Colab version has GPU usage limits
- For large datasets, consider Colab Pro
- Keep the browser tab active during training

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 Paper

This repository contains the implementation related to the following paper:

https://doi.org/10.62520/fujece.1676853
