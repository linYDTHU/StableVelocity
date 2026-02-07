<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

We follow the preprocessing code used in [edm2](https://github.com/NVlabs/edm2). In this code we made a several edits: (1) we removed unncessary parts except preprocessing because this code is only used for preprocessing, (2) we use [-1, 1] range for an input to the stable diffusion VAE (similar to DiT or SiT) unlike edm2 that uses [0, 1] range, and (3) we consider preprocessing to 256x256 resolution (or 512x512 resolution).

After downloading ImageNet, please run the following scripts (please update 256x256 to 512x512 if you want to do experiments on 512x512 resolution);

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images --resolution=256x256 --transform=center-crop-dhariwal \
    --num-workers=8 --batch-size=8
```

```bash
# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=[TARGET_PATH]/images \
    --dest=[TARGET_PATH]/vae-sd --batch-size=16 --num-workers=8
```

Here,`YOUR_DOWNLOAD_PATH` is the directory that you downloaded the dataset, and `TARGET_PATH` is the directory that you will save the preprocessed images and corresponding compressed latent vectors. This directory will be used for your experiment scripts.

## Performance Optimization

The preprocessing scripts have been optimized for better performance:

### Convert Command Optimizations:
- **`--num-workers N`**: Use N parallel workers for image loading and processing (default: 4)
- **`--batch-size N`**: Process N images per batch in each worker (default: 32)
- Full pipeline parallelization including image transforms and PNG encoding
- Optimized image loading and transformation pipelines

### Encode Command Optimizations:
- **`--batch-size N`**: Process N images per VAE encoding batch (default: 8)
- **`--num-workers N`**: Use N parallel workers for image loading (default: 4)  
- Batch processing for VAE encoding to better utilize GPU memory

### Recommended Settings:
- For systems with many CPU cores: `--num-workers=8` or `--num-workers=16`
- For convert with sufficient RAM (>32GB): `--batch-size=16` or `--batch-size=32`
- For convert with limited RAM (<32GB): `--batch-size=4` or `--batch-size=8`
- For encode with large GPU memory: `--batch-size=16` or `--batch-size=32`

### Performance Tips:
1. Use SSD storage for input/output directories for faster I/O
2. Monitor RAM usage and adjust batch size accordingly - the script will show estimated memory usage
3. Set num-workers based on your CPU core count (typically 1-2x cores for stability)
4. If you see "Killed" errors, reduce batch-size or num-workers to use less memory
5. The script automatically limits concurrent batches to prevent excessive memory usage 

## Acknowledgement

This code is mainly built upon preprocessing script from [REPA](https://github.com/sihyun-yu/REPA/tree/main/preprocessing).
