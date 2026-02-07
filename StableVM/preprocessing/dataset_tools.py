# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for creating ZIP/PNG based datasets."""

from dataclasses import dataclass
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union, List
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from encoders import StabilityVAEEncoder

#----------------------------------------------------------------------------

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]
    fname: str  # Add filename for tracking

#----------------------------------------------------------------------------
# Parse a 'M,N' or 'MxN' integer tuple.
# Example: '4x2' returns (4,2)

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

#----------------------------------------------------------------------------

def load_image_fast(fname: str) -> np.ndarray:
    """Load image using PIL with optimization."""
    try:
        # Use PIL.Image.open with optimization
        with PIL.Image.open(fname) as img:
            # Convert to RGB if needed and convert to numpy array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        raise ValueError(f"Could not load image {fname}: {e}")

def process_single_image(args) -> Optional[Tuple[str, np.ndarray, Optional[int]]]:
    """Process a single image - used for multiprocessing."""
    fname, source_dir, labels, transform_func = args
    try:
        # Load image with optimized PIL
        img = load_image_fast(fname)
        
        # Get label
        arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
        label = labels.get(arch_fname)
        
        # Apply transform in worker process for better parallelization
        if transform_func is not None:
            img = transform_func(img)
            if img is None:
                return None
        
        return arch_fname, img, label
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None

def process_image_batch(args) -> List[Optional[Tuple[str, bytes, Optional[int]]]]:
    """Process a batch of images and return PNG bytes - used for multiprocessing."""
    batch_files, source_dir, labels, transform_func = args
    results = []
    
    for fname in batch_files:
        try:
            # Load image
            img = load_image_fast(fname)
            
            # Get label
            arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
            label = labels.get(arch_fname)
            
            # Apply transform
            if transform_func is not None:
                img = transform_func(img)
                if img is None:
                    results.append(None)
                    continue
            
            # Convert to PNG bytes immediately and free memory
            img_pil = PIL.Image.fromarray(img)
            del img  # Free numpy array memory immediately
            
            image_bits = io.BytesIO()
            img_pil.save(image_bits, format='png', compress_level=1, optimize=False)
            del img_pil  # Free PIL image memory immediately
            
            png_bytes = image_bits.getvalue()
            image_bits.close()  # Free BytesIO buffer
            
            results.append((arch_fname, png_bytes, label))
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append(None)
    
    return results

def open_image_folder(source_dir, *, max_images: Optional[int], num_workers: int = 4) -> Tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str): # workaround Path().rglob() slowness
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)
    input_images = input_images[:max_idx]

    # Load labels.
    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    # No labels available => determine from top-level directory names.
    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        # Use multiprocessing for image loading and basic processing
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Prepare arguments for parallel processing
                args_list = [(fname, source_dir, labels, None) for fname in input_images]
                
                # Submit all tasks
                future_to_idx = {executor.submit(process_single_image, args): idx 
                                for idx, args in enumerate(args_list)}
                
                # Collect results in order
                results = [None] * len(args_list)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[idx] = result
                    except Exception as exc:
                        print(f'Image {idx} generated an exception: {exc}')
                
                # Yield results in order
                for result in results:
                    if result is not None:
                        arch_fname, img, label = result
                        yield ImageEntry(img=img, label=label, fname=arch_fname)
        else:
            # Fallback to sequential processing
            for fname in input_images:
                try:
                    img = load_image_fast(fname)
                    arch_fname = arch_fnames[fname]
                    yield ImageEntry(img=img, label=labels.get(arch_fname), fname=arch_fname)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue
    
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def load_image_from_zip(zip_data: bytes) -> np.ndarray:
    """Load image from zip file data."""
    img = PIL.Image.open(io.BytesIO(zip_data)).convert('RGB')
    return np.array(img)

def process_zip_image(args) -> Optional[Tuple[str, np.ndarray, Optional[int]]]:
    """Process a single image from zip - used for multiprocessing."""
    zip_path, fname, labels, transform_func = args
    try:
        with zipfile.ZipFile(zip_path, mode='r') as z:
            with z.open(fname, 'r') as file:
                img = load_image_from_zip(file.read())
        
        # Get label
        label = labels.get(fname)
        
        # Apply transform
        if transform_func is not None:
            img = transform_func(img)
            if img is None:
                return None
        
        return fname, img, label
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None

def open_image_zip(source, *, max_images: Optional[int], num_workers: int = 4) -> Tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)
        input_images = input_images[:max_idx]

        # Load labels.
        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        # Use multiprocessing for zip image processing
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Prepare arguments for parallel processing
                args_list = [(source, fname, labels, None) for fname in input_images]
                
                # Submit all tasks
                future_to_idx = {executor.submit(process_zip_image, args): idx 
                                for idx, args in enumerate(args_list)}
                
                # Collect results in order
                results = [None] * len(args_list)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[idx] = result
                    except Exception as exc:
                        print(f'Image {idx} generated an exception: {exc}')
                
                # Yield results in order
                for result in results:
                    if result is not None:
                        fname, img, label = result
                        yield ImageEntry(img=img, label=label, fname=fname)
        else:
            # Fallback to sequential processing
            with zipfile.ZipFile(source, mode='r') as z:
                for fname in input_images:
                    try:
                        with z.open(fname, 'r') as file:
                            img = load_image_from_zip(file.read())
                        yield ImageEntry(img=img, label=labels.get(fname), fname=fname)
                    except Exception as e:
                        print(f"Error loading {fname}: {e}")
                        continue
    
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

class ImageTransform:
    """Pickleable image transform class for multiprocessing."""
    
    def __init__(self, transform: Optional[str], output_width: Optional[int], output_height: Optional[int]):
        self.transform = transform
        self.output_width = output_width
        self.output_height = output_height
    
    def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
        if self.transform is None:
            return self._scale(img, self.output_width, self.output_height)
        elif self.transform == 'center-crop':
            if self.output_width is None or self.output_height is None:
                raise ValueError('must specify resolution when using center-crop transform')
            return self._center_crop(img, self.output_width, self.output_height)
        elif self.transform == 'center-crop-wide':
            if self.output_width is None or self.output_height is None:
                raise ValueError('must specify resolution when using center-crop-wide transform')
            return self._center_crop_wide(img, self.output_width, self.output_height)
        elif self.transform == 'center-crop-dhariwal':
            if self.output_width is None or self.output_height is None:
                raise ValueError('must specify resolution when using center-crop-dhariwal transform')
            if self.output_width != self.output_height:
                raise ValueError('width and height must match when using center-crop-dhariwal transform')
            return self._center_crop_imagenet(img, self.output_width)
        else:
            raise ValueError(f'unknown transform: {self.transform}')
    
    def _scale(self, img, width, height):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        
        # Use PIL for resizing with optimization
        ww = width if width is not None else w
        hh = height if height is not None else h
        img_pil = PIL.Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img_pil)

    def _center_crop(self, img, width, height):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img_pil = PIL.Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img_pil)

    def _center_crop_wide(self, img, width, height):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img_pil = PIL.Image.fromarray(img, 'RGB')
        img_pil = img_pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img_pil)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    def _center_crop_imagenet(self, arr, image_size: int):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        pil_image = PIL.Image.fromarray(arr)
        while min(*pil_image.size) >= 2 * image_size:
            new_size = tuple(x // 2 for x in pil_image.size)
            assert len(new_size) == 2
            pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(x * scale) for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> ImageTransform:
    """Create a pickleable image transform."""
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
    elif transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
    elif transform == 'center-crop-dhariwal':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        if output_width != output_height:
            raise click.ClickException('width and height must match in --resolution=WxH when using ' + transform + ' transform')
    elif transform is not None:
        raise click.ClickException(f'unknown transform: {transform}')
    
    return ImageTransform(transform, output_width, output_height)

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int], num_workers: int = 4):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images, num_workers=num_workers)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images, num_workers=num_workers)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.group()
def cmdline():
    '''Dataset processing tool for dataset image data conversion and VAE encode/decode preprocessing.'''
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------

@cmdline.command()
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--transform',  help='Input crop/resize mode', metavar='MODE',            type=click.Choice(['center-crop', 'center-crop-wide', 'center-crop-dhariwal']))
@click.option('--resolution', help='Output resolution (e.g., 512x512)', metavar='WxH',  type=parse_tuple)
@click.option('--num-workers', help='Number of worker processes', metavar='INT',        type=int, default=4, show_default=True)
@click.option('--batch-size', help='Batch size for parallel processing', metavar='INT', type=int, default=8, show_default=True)

def convert(
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    num_workers: int,
    batch_size: int
):
    """Convert an image dataset into archive format for training.

    Specifying the input images:

    \b
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, class labels are determined from
    top-level directory names.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    The --transform=center-crop-dhariwal selects a crop/rescale mode that is intended
    to exactly match with results obtained for ImageNet in common diffusion model literature:

    \b
    python dataset_tool.py convert --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \\
        --dest=datasets/img64.zip --resolution=64x64 --transform=center-crop-dhariwal

    Performance options:

    \b
    --num-workers 8                     Use 8 worker processes for parallel processing
    --batch-size 64                     Process 64 images per batch for better efficiency
    """
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    # Get input file list and setup
    if os.path.isdir(source):
        input_images = []
        def _recurse_dirs(root: str):
            with os.scandir(root) as it:
                for e in it:
                    if e.is_file():
                        input_images.append(os.path.join(root, e.name))
                    elif e.is_dir():
                        _recurse_dirs(os.path.join(root, e.name))
        _recurse_dirs(source)
        input_images = sorted([f for f in input_images if is_image_ext(f)])
        max_idx = maybe_min(len(input_images), max_images)
        input_images = input_images[:max_idx]
        
        # Load labels
        labels_dict = dict()
        meta_fname = os.path.join(source, 'dataset.json')
        if os.path.isfile(meta_fname):
            with open(meta_fname, 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels_dict = {x[0]: x[1] for x in data}
        
        if len(labels_dict) == 0:
            arch_fnames = {fname: os.path.relpath(fname, source).replace('\\', '/') for fname in input_images}
            toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
            toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
            if len(toplevel_indices) > 1:
                labels_dict = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}
    else:
        raise click.ClickException('Only directory input supported for optimized processing')

    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    transform_image = make_transform(transform, *resolution if resolution is not None else (None, None))
    dataset_attrs = None
    
    labels = []
    processed_count = 0
    
    # Calculate memory-efficient batch size
    estimated_memory_per_image = 256 * 256 * 3 * 4  # Assume 256x256 RGB, 4 bytes per pixel
    total_concurrent_images = num_workers * batch_size
    estimated_memory_gb = (total_concurrent_images * estimated_memory_per_image) / (1024**3)
    
    print(f"Processing {len(input_images)} images with {num_workers} workers and batch size {batch_size}...")
    print(f"Estimated memory usage: {estimated_memory_gb:.1f}GB for concurrent processing")
    
    if estimated_memory_gb > 16:  # Warn if estimated memory > 16GB
        print(f"WARNING: High memory usage estimated. Consider reducing --batch-size or --num-workers")
    
    # Create batches for parallel processing
    batches = [input_images[i:i + batch_size] for i in range(0, len(input_images), batch_size)]
    
    # Process batches with limited concurrency to control memory
    max_concurrent_batches = min(num_workers, 4)  # Limit concurrent batches
    
    with ProcessPoolExecutor(max_workers=max_concurrent_batches) as executor:
        batch_iter = iter(batches)
        futures = {}
        
        # Submit initial batches
        for _ in range(min(max_concurrent_batches, len(batches))):
            try:
                batch = next(batch_iter)
                batch_idx = len(futures)
                args = (batch, source, labels_dict, transform_image)
                future = executor.submit(process_image_batch, args)
                futures[future] = batch_idx
            except StopIteration:
                break
        
        # Process batches as they complete and submit new ones
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            while futures:
                # Wait for next batch to complete
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)
                
                if not done_futures:
                    # Wait a bit if no futures are done
                    import time
                    time.sleep(0.1)
                    continue
                
                for future in done_futures:
                    batch_idx = futures.pop(future)
                    pbar.update(1)
                    
                    try:
                        batch_results = future.result()
                        
                        for result in batch_results:
                            if result is None:
                                continue
                                
                            arch_fname, png_bytes, label = result
                            
                            # Check image dimensions on first image
                            if dataset_attrs is None:
                                # Decode PNG to check dimensions
                                temp_img = np.array(PIL.Image.open(io.BytesIO(png_bytes)))
                                cur_image_attrs = {'width': temp_img.shape[1], 'height': temp_img.shape[0]}
                                dataset_attrs = cur_image_attrs
                                width = dataset_attrs['width']
                                height = dataset_attrs['height']
                                if width != height:
                                    raise click.ClickException(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
                                if width != 2 ** int(np.floor(np.log2(width))):
                                    raise click.ClickException('Image width/height after scale and crop are required to be power-of-two')
                                del temp_img  # Free memory immediately
                            
                            # Save processed image
                            idx_str = f'{processed_count:08d}'
                            archive_fname_final = f'{idx_str[:5]}/img{idx_str}.png'
                            save_bytes(os.path.join(archive_root_dir, archive_fname_final), png_bytes)
                            labels.append([archive_fname_final, label] if label is not None else None)
                            processed_count += 1
                            
                    except Exception as exc:
                        print(f'Batch {batch_idx} generated an exception: {exc}')
                    
                    # Submit next batch if available
                    try:
                        batch = next(batch_iter)
                        new_batch_idx = len([b for b in batches[:batch_idx+1]]) + len(futures)
                        args = (batch, source, labels_dict, transform_image)
                        new_future = executor.submit(process_image_batch, args)
                        futures[new_future] = new_batch_idx
                    except StopIteration:
                        pass

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

def load_image_batch_for_encode(args):
    """Load a batch of images for VAE encoding - used for multiprocessing."""
    batch_files, source_dir, labels_dict = args
    batch_imgs = []
    batch_labels = []
    batch_fnames = []
    
    for fname in batch_files:
        try:
            # Load image
            if os.path.isdir(source_dir):
                img = load_image_fast(fname)
                arch_fname = os.path.relpath(fname, source_dir).replace('\\', '/')
                label = labels_dict.get(arch_fname)
            else:
                # Handle zip files or processed datasets
                img = load_image_fast(fname)
                arch_fname = os.path.basename(fname)
                label = labels_dict.get(arch_fname)
            
            batch_imgs.append(img)
            batch_labels.append(label)
            batch_fnames.append(arch_fname)
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
    
    return batch_imgs, batch_labels, batch_fnames

def process_batch_vae_encode(batch_data, vae, device='cuda'):
    """Process a batch of images through VAE encoder."""
    batch_imgs, batch_labels, batch_fnames = batch_data
    
    if not batch_imgs:
        return []
    
    # Convert to tensor batch
    batch_tensor = torch.stack([
        torch.tensor(img).permute(2, 0, 1) for img in batch_imgs
    ]).to(device)
    
    # Encode batch
    with torch.no_grad():
        mean_std_batch = vae.encode_pixels(batch_tensor)
    
    # Clear GPU memory immediately
    del batch_tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results = []
    for i, (mean_std, label, fname) in enumerate(zip(mean_std_batch, batch_labels, batch_fnames)):
        results.append((mean_std.cpu(), label, fname))
    
    return results

@cmdline.command()
@click.option('--model-url',  help='VAE encoder model', metavar='URL',                  type=str, default='stabilityai/sd-vae-ft-mse', show_default=True)
@click.option('--source',     help='Input directory or archive name', metavar='PATH',   type=str, required=True)
@click.option('--dest',       help='Output directory or archive name', metavar='PATH',  type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--batch-size', help='Batch size for VAE encoding', metavar='INT',       type=int, default=16, show_default=True)
@click.option('--num-workers', help='Number of worker processes for image loading', metavar='INT', type=int, default=4, show_default=True)

def encode(
    model_url: str,
    source: str,
    dest: str,
    max_images: Optional[int],
    batch_size: int,
    num_workers: int
):
    """Encode pixel data to VAE latents with batch processing and pipeline parallelism."""
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    # Get input file list
    if os.path.isdir(source):
        input_images = []
        def _recurse_dirs(root: str):
            with os.scandir(root) as it:
                for e in it:
                    if e.is_file():
                        input_images.append(os.path.join(root, e.name))
                    elif e.is_dir():
                        _recurse_dirs(os.path.join(root, e.name))
        _recurse_dirs(source)
        input_images = sorted([f for f in input_images if is_image_ext(f)])
        max_idx = maybe_min(len(input_images), max_images)
        input_images = input_images[:max_idx]
        
        # Load labels
        labels_dict = dict()
        meta_fname = os.path.join(source, 'dataset.json')
        if os.path.isfile(meta_fname):
            with open(meta_fname, 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels_dict = {x[0]: x[1] for x in data}
    else:
        raise click.ClickException('Only directory input supported for optimized encode processing')

    # Initialize VAE
    vae = StabilityVAEEncoder(vae_name=model_url, batch_size=batch_size)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    
    print(f"Encoding {len(input_images)} images with batch size {batch_size} and {num_workers} workers...")
    
    # Create batches for image loading
    loading_batch_size = batch_size * 2  # Load more images per batch to keep GPU busy
    image_batches = [input_images[i:i + loading_batch_size] for i in range(0, len(input_images), loading_batch_size)]
    
    labels = []
    processed_count = 0
    
    # Use ThreadPoolExecutor for I/O bound image loading
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading
    
    # Create a queue for loaded batches
    loaded_batch_queue = queue.Queue(maxsize=4)  # Buffer up to 4 loaded batches
    
    def image_loader_worker():
        """Worker thread for loading image batches."""
        with ThreadPoolExecutor(max_workers=num_workers) as loader_executor:
            for batch_idx, image_batch in enumerate(image_batches):
                # Submit loading tasks for this batch
                loading_futures = []
                sub_batch_size = max(1, len(image_batch) // num_workers)
                for i in range(0, len(image_batch), sub_batch_size):
                    sub_batch = image_batch[i:i + sub_batch_size]
                    args = (sub_batch, source, labels_dict)
                    future = loader_executor.submit(load_image_batch_for_encode, args)
                    loading_futures.append(future)
                
                # Collect all sub-batch results
                all_imgs, all_labels, all_fnames = [], [], []
                for future in loading_futures:
                    try:
                        batch_imgs, batch_labels, batch_fnames = future.result()
                        all_imgs.extend(batch_imgs)
                        all_labels.extend(batch_labels)
                        all_fnames.extend(batch_fnames)
                    except Exception as e:
                        print(f"Error in loading batch: {e}")
                
                # Split into VAE batch size chunks and queue them
                for i in range(0, len(all_imgs), batch_size):
                    vae_batch_imgs = all_imgs[i:i + batch_size]
                    vae_batch_labels = all_labels[i:i + batch_size]
                    vae_batch_fnames = all_fnames[i:i + batch_size]
                    
                    if vae_batch_imgs:  # Only queue non-empty batches
                        loaded_batch_queue.put((vae_batch_imgs, vae_batch_labels, vae_batch_fnames))
        
        # Signal end of loading
        loaded_batch_queue.put(None)
    
    # Start image loading in background thread
    loader_thread = threading.Thread(target=image_loader_worker)
    loader_thread.start()
    
    # Process loaded batches with VAE
    total_batches = (len(input_images) + batch_size - 1) // batch_size
    with tqdm(total=len(input_images), desc="Encoding images") as pbar:
        while True:
            batch_data = loaded_batch_queue.get()
            if batch_data is None:  # End signal
                break
            
            # Encode batch with VAE
            try:
                results = process_batch_vae_encode(batch_data, vae)
                
                # Save results
                for mean_std, label, fname in results:
                    idx_str = f'{processed_count:08d}'
                    archive_fname = f'{idx_str[:5]}/img-mean-std-{idx_str}.npy'
                    
                    f = io.BytesIO()
                    np.save(f, mean_std)
                    save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
                    labels.append([archive_fname, label] if label is not None else None)
                    processed_count += 1
                    pbar.update(1)
                    
            except Exception as e:
                print(f"Error encoding batch: {e}")
                # Skip this batch but update progress
                pbar.update(len(batch_data[0]) if batch_data and len(batch_data) > 0 else batch_size)
    
    # Wait for loader thread to finish
    loader_thread.join()
    
    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------