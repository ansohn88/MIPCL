import math
from pathlib import Path, PosixPath
from typing import Union

import numpy as np
import pyvips
from PIL import Image


class WsiProcessError(Exception):
    def __init__(self,
                 message: str
                 ) -> Exception:
        self.message = message


class SlideLoadError(WsiProcessError):
    def __init__(self,
                 message: str
                 ) -> Exception:
        super().__init__(message)


class Slide:

    def __init__(self,
                 path: Union[PosixPath, str]
                 ) -> None:
        if isinstance(path, PosixPath):
            self.path = str(path)
        else:
            self.path = path
        if not Path(path).exists():
            print(SlideLoadError(f"Slide {self.path} not found!"))
            pass
        else:
            self.filestem = Path(path).stem
            self.filename = Path(path).name
            self.fileparent = Path(path).parent

            self.cytowsi = self.load_slide()

    def load_slide(self):
        num_of_pages = int(pyvips.Image.new_from_file(
            self.path).get("n-pages"))
        cytowsi = [
            pyvips.Image.new_from_file(
                self.path, page=n
            ) for n in range(2, num_of_pages)
        ]
        return cytowsi

    def vips_2_nparr(self) -> np.ndarray:
        self.cytowsi_np = np.stack(
            [Slide.numpy_from_vips(self.cytowsi[page])
             for page in range(len(self.cytowsi))],
            axis=3
        )
        return self.cytowsi_np

    def vips_2_nparr_z(self,
                       which_z: int) -> np.ndarray:
        return Slide.numpy_from_vips(self.cytowsi[which_z])

    def __str__(self):
        return f"""
        Slide path: {self.path}\n
        Slide height: {self.cytowsi[0].height}\n
        Slide width: {self.cytowsi[0].width}\n
        Slide channels: {self.cytowsi[0].bands}\n
        Slide num of z-stack: {len(self.cytowsi)}
        """

    def crop_region(self,
                    x: int,
                    y: int,
                    patch_size: int,
                    which_z: int
                    ) -> Image:
        crop = self.cytowsi[which_z].crop(
            x*patch_size, y*patch_size, patch_size, patch_size)
        patch = np.ndarray(
            buffer=crop.write_to_memory(),
            dtype=np.uint8,
            shape=[crop.height, crop.width, crop.bands]
        )
        return Image.fromarray(patch)

    def fetch_region(self,
                     x: int,
                     y: int,
                     patch_size: int,
                     which_z: int) -> Image:
        reg = pyvips.Region.new(self.cytowsi[which_z])
        patch = reg.fetch(patch_size * x, patch_size *
                          y, patch_size, patch_size)
        patch = np.ndarray(
            buffer=patch,
            dtype=np.uint8,
            shape=(patch_size, patch_size, 3)
        )
        # return patch
        return Image.fromarray(patch)

    @staticmethod
    def vips_from_memory(arr: np.ndarray,
                         return_z: bool,
                         which_z: int = 6
                         ) -> pyvips.Image:
        if return_z:
            h = arr[:, :, :, which_z].shape[0]
            w = arr[:, :, :, which_z].shape[1]
            c = arr[:, :, :, which_z].shape[2]
            linear = arr[:, :, :, which_z]\
                .reshape(int(h) * int(w) * int(c))
            linear = np.ascontiguousarray(linear)
            vi = pyvips.Image.new_from_memory(
                linear.data,
                int(h),
                int(w),
                int(c),
                'uchar'
            )
        else:
            h = arr.shape[0]
            w = arr.shape[1]
            c = arr.shape[2]
            linear = arr.reshape(int(h) * int(w) * int(c))
            linear = np.ascontiguousarray(linear)
            vi = pyvips.Image.new_from_memory(
                linear.data,
                int(h),
                int(w),
                int(c),
                'uchar'
            )
        return vi

    @staticmethod
    def numpy_from_vips(vi) -> np.ndarray:
        return np.ndarray(
            buffer=vi.write_to_memory(),
            dtype=np.uint8,
            shape=[vi.height, vi.width, vi.bands]
        )

    def show_slide(self,
                   which_z: int,
                   scale_factor: int
                   ) -> Image:
        thumb = self.get_thumbnail(which_z, scale_factor)
        thumb = Image.fromarray(Slide.numpy_from_vips(thumb))
        thumb.show()

    def export_z_plane_tiff(self,
                            which_z: int,
                            save_as: str) -> None:
        panc_cyto_stack = [
            pyvips.Image.new_from_file(self.path, page=n)
            for n in range(2, 15)
        ]
        z_plane = panc_cyto_stack[which_z]
        z_plane.write_to_file(save_as)

    def export_thumbnail(self,
                         which_z: int,
                         scale_factor: int,
                         save_as: str,
                         ) -> None:
        thumb = self.get_thumbnail(which_z, scale_factor)
        thumb.pngsave(save_as)

    def get_thumbnail(self,
                      which_z: int,
                      scale_factor: int) -> pyvips.Image:
        if not which_z < len(self.cytowsi):
            raise ValueError("Z-stack index out of range")

        new_w = math.floor(self.cytowsi[which_z].width / scale_factor)
        new_h = math.floor(self.cytowsi[which_z].height / scale_factor)
        thumbnail = self.cytowsi[which_z].thumbnail_image(new_w, height=new_h)
        return thumbnail

    def get_thumbnail_np(self,
                         which_z: int,
                         scale_factor: int
                         ) -> pyvips.Image:
        if not which_z < len(self.cytowsi):
            raise ValueError("Z-stack index out of range.")

        vi = self.get_thumbnail(which_z=which_z, scale_factor=scale_factor)
        arr = self.numpy_from_vips(vi=vi)

        return arr
