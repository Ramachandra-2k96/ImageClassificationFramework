# CompactVisionNet Model Training and Fine-tuning Guide

This guide explains how to use and fine-tune the CompactVisionNet model for image classification tasks.

## Project Structure
```
/project_root
├── DATASETS/   # Upload your dataset here
│   ├── train/                 # Training data
│   └── val/                   # Validation data
├── BASE_MODELS/
│   └── best_model.pth          # Core model architecture (DO NOT MODIFY)
├── Classify/
│   ├── Classify.py
|   ├──  Test.py    
│   └──  FineTune.py
└── app.py    # Finetuning script [This one you should modify and run to get your model]
└── evaluation.py  # Evaluation script[Test your finetuned model]
```

## Prerequisites
- Python 3.10+
- PyTorch 1.13+
- torchvision
- numpy
- pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Organize your dataset in the following structure:
```
DTASETS/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    └── class2/
```

2. Images should be in JPG/PNG format
3. Each class should have its own directory
4. Recommended minimum: 100 images per class

## Configuration Setup

### Files You Should NOT Modify
- `models/Classify.py` - Core architecture
- `utils/app.py` - Data loading logic
- `utils/evaluation.py` - Evaluation logic

## Fine-tuning Steps

1. **Prepare Your Data**
   - Clean your dataset
   - Split into train/val sets
   - Place in appropriate directories

2. **Configure Training**
   - Modify parameters according to your needs
   - Important parameters:
     - `num_classes`: Number of target classes
     - `freeze_backbone`: Set to `true` for transfer learning
     - `batch_size`: Adjust based on your GPU memory

3. **Start Training**
```bash
python app.py
```

4. **Monitor Training**
   - Training logs will be displayed in the terminal
   - Checkpoints saved in 'Heavy_models' folder
   - Monitor using TensorBoard:
```bash
tensorboard --logdir logs/
```

## Best Practices

1. **Transfer Learning**
   - Start with `freeze_backbone: true`
   - Train for 10-20 epochs
   - Set `freeze_backbone: false`
   - Continue training for full duration

2. **Learning Rate**
   - Start with 0.001
   - Use learning rate scheduler
   - Reduce on plateau if validation loss stagnates

## Troubleshooting

1. **Out of Memory**
   - Reduce batch size
   - Reduce image size
   - Use mixed precision training

2. **Poor Convergence**
   - Check class balance
   - Adjust learning rate
   - Increase training epochs
   - Verify data quality

3. **Overfitting**
   - Increase dropout
   - Add data augmentation
   - Reduce model capacity
   - Implement early stopping

## Model Export

After training, models are saved in `Heavy_models/`. To export for inference:

## License
The code is provided under the MIT License. See the LICENSE file for details.

## Citation
If you use this model in your research, please cite:

```
[Your citation information]
```

For questions and support, please open an issue on the GitHub repository.
