# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import TypeAlias, Union
from urllib.parse import urlparse

import httpx
import pybase64
import torch
from PIL import Image

from .http_client import get_http_client

logger = logging.getLogger(__name__)

# Image output can be either PIL Image or Tensor (from nvimgcodec)
ImageOutput: TypeAlias = Union[Image.Image, torch.Tensor]

# Thread-local storage for nvimgcodec decoders
_thread_local = threading.local()

# Lazy import for nvimgcodec
_nvimgcodec = None

# Global thread pool for image decoding operations
# Default to 8 workers, configurable via DYN_IMAGE_DECODE_WORKERS env var
_IMAGE_DECODE_WORKERS = int(os.environ.get("DYN_IMAGE_DECODE_WORKERS", 8))
_decode_thread_pool = ThreadPoolExecutor(
    max_workers=_IMAGE_DECODE_WORKERS,
    thread_name_prefix="image_decode_",
)


def _get_nvimgcodec():
    """Lazy import nvimgcodec to avoid import errors if not installed."""
    global _nvimgcodec
    if _nvimgcodec is None:
        from nvidia import nvimgcodec
        _nvimgcodec = nvimgcodec
    return _nvimgcodec


def get_decoder():
    """Get or create a thread-local nvimgcodec decoder instance."""
    if not hasattr(_thread_local, "decoder"):
        nvimgcodec = _get_nvimgcodec()
        _thread_local.decoder = nvimgcodec.Decoder()
        logger.info("nvimgcodec decoder initialized for thread")
    return _thread_local.decoder


class ImageLoader:
    CACHE_SIZE_MAXIMUM = 8

    def __init__(
        self,
        cache_size: int = CACHE_SIZE_MAXIMUM,
        http_timeout: float = 30.0,
        use_nvimgcodec: bool = True,
        image_mode: str = "RGB",
    ):
        """
        Initialize the ImageLoader.

        Args:
            cache_size: Maximum number of images to cache
            http_timeout: Timeout for HTTP requests
            use_nvimgcodec: If True, use nvimgcodec for GPU-accelerated decoding
                           (returns 4D torch.Tensor). If False, use PIL (returns Image.Image)
            image_mode: Target image mode for PIL conversion (default: "RGB")
        """
        self._http_timeout = http_timeout
        self._use_nvimgcodec = use_nvimgcodec
        self._image_mode = image_mode
        self._image_cache: dict[str, ImageOutput] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)

    def _decode_with_nvimgcodec(self, data: bytes) -> torch.Tensor:
        """
        Decode image bytes using nvimgcodec for GPU-accelerated decoding.

        Args:
            data: Raw image bytes

        Returns:
            torch.Tensor in NCHW format (4D) on CUDA device.
            Shape: (1, C, H, W) - batch dimension added so vLLM treats it as
            a batch of images, not as embeddings.
        """
        decoder = get_decoder()
        decoded = decoder.decode(data)

        device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.as_tensor(decoded, device=device)
        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)
        # Add batch dimension: CHW -> NCHW (1, C, H, W)
        # This is critical: 3D tensors are interpreted as embeddings by vLLM,
        # but 4D tensors are interpreted as a batch of images.
        tensor = tensor.unsqueeze(0)

        return tensor

    def _decode_with_pil(self, data: bytes) -> Image.Image:
        """
        Decode image bytes using PIL.

        Args:
            data: Raw image bytes

        Returns:
            PIL Image converted to the target image mode
        """
        image = Image.open(BytesIO(data))

        # Validate image format
        if image.format not in ("JPEG", "PNG", "WEBP", "GIF"):
            raise ValueError(f"Unsupported image format: {image.format}")

        # Convert to target mode
        if image.mode != self._image_mode:
            image = image.convert(self._image_mode)

        return image

    async def _fetch_image_bytes(self, image_url: str) -> bytes:
        """
        Fetch image bytes from a URL or data URI.

        Args:
            image_url: URL (http/https) or data URI (data:image/...;base64,...)

        Returns:
            Raw image bytes
        """
        parsed_url = urlparse(image_url)

        if parsed_url.scheme == "data":
            # Parse data URL format: data:[<media type>][;base64],<data>
            if not parsed_url.path.startswith("image/"):
                raise ValueError("Data URL must be an image type")

            # Split the path into media type and data
            media_type, data = parsed_url.path.split(",", 1)
            if ";base64" not in media_type:
                raise ValueError("Data URL must be base64 encoded")

            try:
                # Use pybase64 for faster base64 decoding
                return pybase64.b64decode(data, validate=True)
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding: {e}")

        elif parsed_url.scheme in ("http", "https"):
            http_client = get_http_client(self._http_timeout)

            response = await http_client.get(image_url)
            response.raise_for_status()

            if not response.content:
                raise ValueError("Empty response content from image URL")

            return response.content

        else:
            raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")

    async def load_image(self, image_url: str) -> ImageOutput:
        """
        Load an image from a URL or data URI.

        Args:
            image_url: URL (http/https) or data URI (data:image/...;base64,...)

        Returns:
            torch.Tensor in NCHW format (if use_nvimgcodec=True) or PIL Image
        """
        parsed_url = urlparse(image_url)

        # For HTTP(S) URLs, check cache first
        if parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                return self._image_cache[image_url_lower]

        try:
            # Fetch image bytes
            image_bytes = await self._fetch_image_bytes(image_url)

            # Decode the image using thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            if self._use_nvimgcodec:
                # nvimgcodec decoding (GPU-accelerated, returns 4D tensor)
                # Offload to thread pool to avoid blocking the event loop
                image_result = await loop.run_in_executor(
                    _decode_thread_pool, self._decode_with_nvimgcodec, image_bytes
                )
            else:
                # PIL decoding (CPU-bound, offload to thread pool)
                image_result = await loop.run_in_executor(
                    _decode_thread_pool, self._decode_with_pil, image_bytes
                )

            # Cache HTTP(S) URLs
            if parsed_url.scheme in ("http", "https"):
                image_url_lower = image_url.lower()
                # Cache the image for future use, and evict the oldest image if full
                if self._cache_queue.full():
                    oldest_image_url = await self._cache_queue.get()
                    del self._image_cache[oldest_image_url]

                self._image_cache[image_url_lower] = image_result
                await self._cache_queue.put(image_url_lower)

            return image_result

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")
