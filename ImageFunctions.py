# -*- coding: utf-8 -*-

__author__ = "Alexander Bohn"

import sys

import numpy as np
import photutils as pu
import scipy as sc
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift, rotate
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import register_translation
from skimage.transform import rescale

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import memory_frames, progress
from pynpoint.util.image import rotate_images, scale_image


class StarAlignmentModule(ProcessingModule):
    """
    Module to align the images with a cross-correlation in Fourier space.
    """

    def __init__(self,
                 name_in="star_align",
                 image_in_tag="im_arr",
                 ref_image_in_tag=None,
                 image_out_tag="im_arr_aligned",
                 ref_index=0,
                 interpolation="spline",
                 center=None,
                 center_size=None,
                 accuracy=10,
                 resize=None,
                 num_references=10):
        """
        Constructor of StarAlignmentModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry with the stack of images that is read as
                             input.
        :type image_in_tag: str
        :param ref_image_in_tag: Tag of the database entry with the reference image(s)
                                 that are read as input. If it is set to None, a random
                                 subsample of *num_references* elements of *image_in_tag*
                                 is taken as reference image(s)
        :type ref_image_in_tag: str
        :param image_out_tag: Tag of the database entry with the images that are written as
                              output.
        :type image_out_tag: str
        :param ref_index: Index of the image of the reference frames that should be used for
                          the alignment. All images are used if set to None.
        :type ref_index: int, None
        :param interpolation: Type of interpolation that is used for shifting the images (spline,
                              bilinear, or fft).
        :type interpolation: str
        :param accuracy: Upsampling factor for the cross-correlation. Images will be registered
                         to within 1/accuracy of a pixel.
        :type accuracy: float
        :param resize: Scaling factor for the up/down-sampling before the images are shifted.
        :type resize: float
        :param num_references: Number of reference images for the cross-correlation.
        :type num_references: int

        :return: None
        """

        super(StarAlignmentModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if ref_image_in_tag is not None:
            self.m_ref_image_in_port = self.add_input_port(ref_image_in_tag)
        else:
            self.m_ref_image_in_port = None

        self.m_ref_index = ref_index
        self.m_interpolation = interpolation
        self.m_center = center
        self.m_center_size = center_size
        self.m_accuracy = accuracy
        self.m_resize = resize
        self.m_num_references = num_references

    def run(self):
        """
        Run method of the module. Applies a cross-correlation of the input images with respect to
        a stack of reference images, rescales the image dimensions, and shifts the images to a
        common center.

        :return: None
        """

        if self.m_ref_image_in_port is not None:
            im_dim = self.m_ref_image_in_port.get_ndim()

            if im_dim == 3:
                if self.m_ref_image_in_port.get_shape()[0] > self.m_num_references:
                    if self.m_ref_index != None:
                        ref_images = np.array([self.m_ref_image_in_port[self.m_ref_index], ])
                    else:
                        ref_images = self.m_ref_image_in_port[np.sort(
                            np.random.choice(self.m_ref_image_in_port.get_shape()[0],
                                             self.m_num_references,
                                             replace=False)), :, :]

                else:
                    ref_images = self.m_ref_image_in_port.get_all()
                    self.m_num_references = self.m_ref_image_in_port.get_shape()[0]

            elif im_dim == 2:
                ref_images = np.array([self.m_ref_image_in_port.get_all(), ])
                self.m_num_references = 1

            else:
                raise ValueError("reference Image needs to be 2 D or 3 D.")

        else:
            random = np.random.choice(self.m_image_in_port.get_shape()[0],
                                      self.m_num_references,
                                      replace=False)
            sort = np.sort(random)
            ref_images = self.m_image_in_port[sort, :, :]

        def _crop_image(image_in,
                        center,
                        size):
            """
            Simple function to crop an image around a specified position.

            :param image_in:
            :param center: Tuple (x0, y0) with the new image center. Python indexing starts at 0. The
                           center of the input images will be used when *center* is set to *None*.
            :type center: tuple, int
            :param size: New image size (arcsec). The same size will be used for both image dimensions.
            :type size: int

            :return: cropped image
            """
            im = image_in.copy()

            x_off = center[0] - int(size / 2)
            y_off = center[1] - int(size / 2)

            return im[y_off: size + y_off,
                      x_off: size + x_off]

        def _align_image(image_in):
            offset = np.array([0., 0.])

            for i in range(self.m_num_references):
                tmp_offset, _, _ = register_translation(_crop_image(ref_images[i,:,:], self.m_center, self.m_center_size),
                                                        _crop_image(image_in, self.m_center, self.m_center_size),
                                                        upsample_factor=self.m_accuracy)
                offset += tmp_offset

            offset /= float(self.m_num_references)
            if self.m_resize is not None:
                offset *= self.m_resize

            if self.m_resize is not None:
                sum_before = np.sum(image_in)
                tmp_image = rescale(image=np.asarray(image_in, dtype=np.float64),
                                    scale=(self.m_resize, self.m_resize),
                                    order=5,
                                    mode="reflect")
                sum_after = np.sum(tmp_image)

                # Conserve flux because the rescale function normalizes all values to [0:1].
                tmp_image = tmp_image*(sum_before/sum_after)

            else:
                tmp_image = image_in

            if self.m_interpolation == "spline":
                tmp_image = shift(tmp_image, offset, order=5)

            elif self.m_interpolation == "bilinear":
                tmp_image = shift(tmp_image, offset, order=1)

            elif self.m_interpolation == "fft":
                tmp_image_spec = fourier_shift(np.fft.fftn(tmp_image), offset)
                tmp_image = np.fft.ifftn(tmp_image_spec).real

            else:
                raise ValueError("Interpolation should be spline, bilinear, or fft.")

            return tmp_image

        self.apply_function_to_images(_align_image,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running StarAlignmentModule...")

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        if self.m_resize is not None:
            pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
            self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_resize)

        if self.m_resize is None:
            history = "cross-correlation, no upsampling"
        else:
            history = "cross-correlation, upsampling factor =" + str(self.m_resize)
        self.m_image_out_port.add_history("PSF alignment", history)
        self.m_image_out_port.close_port()


class UnsharpMaskModule(ProcessingModule):

    def __init__(self,
                 name_in="apply_unsharp_mask",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_unsharp",
                 kernel_size=5):

        super(UnsharpMaskModule, self).__init__(name_in)

        # Variables
        self.m_kernel_size = kernel_size

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        def apply_unsharp_mask(image_in,
                               kernel_size):

            unsharp = gaussian_filter(image_in,
                                      sigma=kernel_size)

            return image_in-unsharp

        self.apply_function_to_images(apply_unsharp_mask,
                                      image_in_port=self.m_image_in_port,
                                      image_out_port=self.m_image_out_port,
                                      message="Running UnsharpMaskModule...",
                                      func_args=(self.m_kernel_size,))

        self.m_image_out_port.add_history("Unsharp Masking", "Gaussian kerne size: %s"%self.m_kernel_size)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()

        sys.stdout.flush()


class CreateMedianModule(ProcessingModule):

    def __init__(self,
                 name_in="create_median",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_median"):

        super(CreateMedianModule, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        sys.stdout.write("Running CreateMedianModule...")
        sys.stdout.flush()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        data = self.m_image_in_port.get_all()

        self.m_image_out_port.set_all(np.median(data,axis=0))
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Image Stacking", "Creating median")
        self.m_image_out_port.close_port()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()


class MaskNegativePixelsModule(ProcessingModule):

    def __init__(self,
                 threshold,
                 damping=1.,
                 name_in="mask_negative_pixels",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_masked"):

        super(MaskNegativePixelsModule, self).__init__(name_in)

        # Variables
        self.m_threshold = threshold
        self.m_damping = damping

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        sys.stdout.write("Running MaskNegativePixelsModule...")
        sys.stdout.flush()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        data = self.m_image_in_port.get_all()
        data_masked = data.copy()
        data_masked[data<self.m_threshold] = 0.

        self.m_image_out_port.set_all(data_masked * self.m_damping)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Mask data", "Masked pixel values smaller than %s"%self.m_threshold)
        self.m_image_out_port.close_port()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()


class MaskNegativePixelsModule_2(ProcessingModule):

    def __init__(self,
                 threshold,
                 damping=1.,
                 name_in="mask_negative_pixels",
                 image_in_tag="im_arr",
                 res_in_tag="res_median",
                 parang_tag="parang",
                 image_out_tag="im_arr_masked"):

        super(MaskNegativePixelsModule_2, self).__init__(name_in)

        # Variables
        self.m_threshold = threshold
        self.m_damping = damping

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_res_in_port = self.add_input_port(res_in_tag)
        self.m_parang_in_port = self.add_input_port(parang_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        sys.stdout.write("Running MaskNegativePixelsModule_2...")
        sys.stdout.flush()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        res_median = self.m_res_in_port.get_all()

        mask = np.ones_like(res_median)
        mask[res_median<self.m_threshold] = 0.

        data_masked = self.m_image_in_port.get_all()
        parang = self.m_parang_in_port.get_attribute("PARANG")

        for i in range(data_masked.shape[0]):
            tmp_mask = rotate(mask,parang[i],
                              reshape=False)

            data_masked[i,] = tmp_mask * data_masked[i,]

        self.m_image_out_port.set_all(data_masked * self.m_damping)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Mask data", "Masked pixel values smaller than %s"%self.m_threshold)
        self.m_image_out_port.close_port()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()

class SubtractCubesModule(ProcessingModule):

    def __init__(self,
                 name_in="subtract_cubes",
                 image_in_tag_1="im_arr_1",
                 image_in_tag_2="im_arr_2",
                 image_out_tag="im_arr_sub"):

        super(SubtractCubesModule, self).__init__(name_in)

        # Tags
        self.m_image_in_tag_1 = image_in_tag_1
        self.m_image_in_tag_2 = image_in_tag_2

        # Ports
        self.m_image_in_port_1 = self.add_input_port(image_in_tag_1)
        self.m_image_in_port_2 = self.add_input_port(image_in_tag_2)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        sys.stdout.write("Running SubtractCubesModule...")
        sys.stdout.flush()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        data_1 = self.m_image_in_port_1.get_all()
        data_2 = self.m_image_in_port_2.get_all()

        self.m_image_out_port.set_all(data_1-data_2)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port_1)
        self.m_image_out_port.add_history_information("Subtract images", "%s - %s"%(self.m_image_in_tag_1,self.m_image_in_tag_2))
        self.m_image_out_port.close_port()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()


class ScaleCubeModule(ProcessingModule):

    def __init__(self,
                 name_in="subtract_cubes",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled"):

        super(ScaleCubeModule, self).__init__(name_in)

        # Tags
        self.m_image_in_tag = image_in_tag

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        sys.stdout.write("Running ScaleCubeModule...")
        sys.stdout.flush()

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        data = self.m_image_in_port.get_all()
        norm = self.m_image_in_port.get_attribute("norm")

        data_scaled = data.copy()

        for i, n in enumerate(norm):
            data_scaled[i,] = data[i,]*n

        self.m_image_out_port.set_all(data_scaled)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_history_information("Scale images", "Image norm")
        self.m_image_out_port.close_port()

        sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()


class Subtract180Module(ProcessingModule):

    def __init__(self,
                 name_in="subtract_180_image",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_sub_180"):

        super(Subtract180Module, self).__init__(name_in)

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        im_data = self.m_image_in_port.get_all()

        def sub_180(image_in):


            # rotate image data
            im_rot_180 = sc.ndimage.rotate(input=image_in,
                                           angle=180,
                                           reshape=False)

            return image_in-im_rot_180

        # save results
        self.apply_function_to_images(sub_180,
                                      image_in_port=self.m_image_in_port,
                                      image_out_port=self.m_image_out_port,
                                      message="Running Subtract180Module...",
                                      func_args=())

        self.m_image_out_port.add_history("Image subtraction", "Subtracting 180 deg rotated image")
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()

class SmoothImagesModule(ProcessingModule):

    def __init__(self,
                 name_in="smooth_images",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_smoothed",
                 kernel_size=5):

        super(SmoothImagesModule, self).__init__(name_in)

        # Variables
        self.m_kernel_size = kernel_size

        # Ports
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        sys.stdout.write("Running SmoothImagesModule...")
        sys.stdout.flush()

        def apply_unsharp_mask(image_in,
                               kernel_size):

            unsharp = gaussian_filter(image_in,
                                      sigma=kernel_size)

            return unsharp

        self.apply_function_to_images(apply_unsharp_mask,
                                      image_in_port=self.m_image_in_port,
                                      image_out_port=self.m_image_out_port,
                                      message="Running SmoothImagesModule...",
                                      func_args=(self.m_kernel_size,))

        self.m_image_out_port.add_history_information("Image smoothing", "Gaussian kerne size: %s"%self.m_kernel_size)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()

        sys.stdout.flush()


class AnnulusBackgroundSubtractionModule(ProcessingModule):

    def __init__(self,
                 annulus_size,
                 name_in="background_subtraction",
                 image_in_tag="im_arr",
                 image_out_tag="bg_cleaned_arr"):

        super(AnnulusBackgroundSubtractionModule, self).__init__(name_in)

        # Ports

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Parameter
        self.m_annulus_size = annulus_size

    def run(self):

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        def subtract_background(image_in,
                                annulus_size):

            edge_offset = 5
            im_shape=image_in.shape

            # Define circular sky annulus for background subtraction
            annulus = pu.CircularAnnulus(positions=(im_shape[0] / 2. - .5, im_shape[1] / 2. -.5),
                                         r_in=np.min(im_shape) / 2. -.5 - annulus_size - edge_offset,
                                         r_out=np.min(im_shape) / 2. - .5 - edge_offset)

            # Determine median flux per pixel inside annulus
            annulus_mask = annulus.to_mask(method="center")
            weighted_data = annulus_mask.multiply(image_in)
            bg_flux = np.median(weighted_data[weighted_data != 0])

            return image_in - bg_flux

        # align all Science data
        self.apply_function_to_images(func=subtract_background,
                                      image_in_port=self.m_image_in_port,
                                      image_out_port=self.m_image_out_port,
                                      message="Running AnnulusBackgroundSubtractionModule...",
                                      func_args=(self.m_annulus_size,))

        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        history = "annulus method"
        self.m_image_out_port.add_history("Background subtraction",history)

        self.m_image_out_port.close_port()

        # sys.stdout.write(" [DONE]\n")
        sys.stdout.flush()


class MultiplyImagesModule(ProcessingModule):
    """
    Module for multiplying two sets of images.
    """

    def __init__(self,
                 image_in_tags,
                 name_in="multiply_images",
                 image_out_tag="im_arr_mult",
                 scaling=1.):
        """
        Constructor of AddImagesModule.

        :param image_in_tags: Tuple with two tags of the database entry that are read as input.
        :type image_in_tags: (str, str)
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_out_tag: Tag of the database entry with the added images that are written as
                              output. Should be different from *image_in_tags*.
        :type image_out_tag: str
        :param scaling: Additional scaling factor.
        :type scaling: float

        :return: None
        """

        super(MultiplyImagesModule, self).__init__(name_in=name_in)

        self.m_image_in1_port = self.add_input_port(image_in_tags[0])
        self.m_image_in2_port = self.add_input_port(image_in_tags[1])
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_scaling = scaling

    def run(self):
        """
        Run method of the module. Multiply the images from the two database tags.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        # if self.m_image_in1_port.get_shape() != self.m_image_in2_port.get_shape():
        #     raise ValueError("The shape of the two input tags have to be equal.")

        nimages = self.m_image_in1_port.get_shape()[0]
        memory = self._m_config_port.get_attribute("MEMORY")

        frames = memory_frames(memory, nimages)

        image2 = np.squeeze(self.m_image_in2_port.get_all())

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running MultiplyImagesModule...")

            images1 = self.m_image_in1_port[frames[i]:frames[i+1], ]

            self.m_image_out_port.append(self.m_scaling*(images1*image2))

        sys.stdout.write("Running MultiplyImagesModule... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history("Images multiplied", "")
        self.m_image_out_port.copy_attributes(self.m_image_in1_port)
        self.m_image_out_port.close_port()


class DerotateAndOrStackMaskModule(ProcessingModule):
    """
    Module for derotating and/or stacking (i.e., taking the median or average) of the images.
    """

    def __init__(self,
                 threshold,
                 name_in="derotate_stack",
                 image_in_tag="im_arr",
                 image_out_tag="im_stack",
                 derotate=True,
                 stack=None,
                 extra_rot=0.):
        """
        Constructor of DerotateAndOrStackModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. The output is
                              either 2D (*stack=False*) or 3D (*stack=True*).
        :type image_out_tag: str
        :param derotate: Derotate the images with the PARANG attribute.
        :type derotate: bool
        :param stack: Type of stacking applied after optional derotation ("mean", "median",
                      or None for no stacking).
        :type stack: str
        :param extra_rot: Additional rotation angle of the images in clockwise direction (deg).
        :type extra_rot: float

        :return: None
        """

        super(DerotateAndOrStackMaskModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_threshold = threshold
        self.m_derotate = derotate
        self.m_stack = stack
        self.m_extra_rot = extra_rot

    def run(self):
        """
        Run method of the module. Uses the PARANG attributes to derotate the images (if *derotate*
        is set to True) and applies an optional mean or median stacking afterwards.

        :return: None
        """

        def _initialize(ndim, npix):
            if ndim == 2:
                nimages = 1
            elif ndim == 3:
                nimages = self.m_image_in_port.get_shape()[0]

            if self.m_stack == "median":
                frames = [0, nimages]
            else:
                frames = memory_frames(memory, nimages)

            if self.m_stack == "mean":
                im_tot = np.zeros((npix, npix))
            else:
                im_tot = None

            return nimages, frames, im_tot

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output port should have a different tag.")

        memory = self._m_config_port.get_attribute("MEMORY")

        if self.m_derotate:
            parang = self.m_image_in_port.get_attribute("PARANG")

        ndim = self.m_image_in_port.get_ndim()
        npix = self.m_image_in_port.get_shape()[1]

        nimages, frames, im_tot = _initialize(ndim, npix)

        for i, _ in enumerate(frames[:-1]):
            progress(i, len(frames[:-1]), "Running DerotateAndOrStackMaskModule...")

            images = self.m_image_in_port[frames[i]:frames[i+1], ]

            if self.m_derotate:
                images[np.abs(images) < self.m_threshold] = 0.

                angles = -parang[frames[i]:frames[i+1]]+self.m_extra_rot
                images = rotate_images(images, angles)

                images[np.abs(images) < self.m_threshold] = np.nan

            if self.m_stack is None:
                if ndim == 2:
                    self.m_image_out_port.set_all(images)
                elif ndim == 3:
                    self.m_image_out_port.append(images, data_dim=3)

        sys.stdout.write("Running DerotateAndOrStackMaskModule... [DONE]\n")
        sys.stdout.flush()

        if self.m_stack == "mean":
            tmp_mean = np.nanmean(images, axis=0)
            mean = np.nan_to_num(tmp_mean)
            self.m_image_out_port.set_all(mean)

        elif self.m_stack == "median":
            tmp_median = np.nanmedian(images, axis=0)
            median = np.nan_to_num(tmp_median)
            self.m_image_out_port.set_all(median)

        if self.m_derotate or self.m_stack is not None:
            self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.close_port()

class CorrectDistortionModule_SPHERE(ProcessingModule):
    """
    Module for correcting the distortion in y direction in SPHERE data
    """

    def __init__(self,
                 name_in="derotate_stack",
                 image_in_tag="im_arr",
                 image_out_tag="im_stack"):
        """
        Constructor of CorrectDistortionModule_SPHERE.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str

        :return: None
        """

        super(CorrectDistortionModule_SPHERE, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run method of the module. Crops the images in y direction to 967 lines.
        Scales y axis by 1.0062 according to SPHERE manual

        :return: None
        """

        def _remove_lines(image_in,
                          lines):
            shape_in = image_in.shape

            return image_in[int(lines[2]):shape_in[0]-int(lines[3]),
                            int(lines[0]):shape_in[1]-int(lines[1])]

        def _add_lines(image_in,
                       lines):
            image_out = np.zeros((1024, 2048))

            image_out[int(lines[2]):int(image_out.shape[0] - lines[3]),
                      int(lines[0]):int(image_out.shape[1] - lines[1])] = image_in

            return image_out

        def _correct_distortion(image_in):

            im_removed = _remove_lines(image_in,
                                       [0, 0, (1024 - 968) / 2, (1024 - 968) / 2])

            im_scaled = scale_image(im_removed,scaling_x=1.,scaling_y=1.0062)

            im_corrected = _add_lines(im_scaled,
                                      [0, 0, 25, 25])

            return im_corrected

        self.apply_function_to_images(_correct_distortion,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running CorrectDistortionModule_SPHERE...")

        # sys.stdout.write("Running CorrectDistortionModule_SPHERE... [DONE]\n")
        sys.stdout.flush()

        self.m_image_out_port.add_history("Corrected image distortion", "Multiplied y axis by 1.0062")
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        self.m_image_out_port.close_port()