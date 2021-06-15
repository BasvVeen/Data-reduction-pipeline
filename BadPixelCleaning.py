"""
Pypline Modules for the detection and interpolation of bad pixels
"""
import numpy as np
from photutils.aperture import RectangularAperture

from pynpoint.core.processing import ProcessingModule

__author__ = "Alexander Bohn"
# from numba import jit

class BadPixelMapModule_IRDIS(ProcessingModule):
    """
    Module to create a bad pixel map from the dark frames and flat fields.
    """

    def __init__(self,
                 name_in="bad_pixel_map",
                 dark_in_tag="dark",
                 flat_in_tag="flat",
                 bp_map_out_tag="bp_map",
                 dark_threshold=0.2,
                 flat_threshold=0.2):
        """
        Constructor of BadPixelMapModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param dark_in_tag: Tag of the database entry with the dark frames that are read as input.
                            Not read if set to None.
        :type dark_in_tag: str
        :param flat_in_tag: Tag of the database entry with the flat fields that are read as input.
                            Not read if set to None.
        :type flat_in_tag: str
        :param bp_map_out_tag: Tag of the database entry with the bad pixel map that is written as
                               output.
        :type bp_map_out_tag: str
        :param dark_threshold: Fractional threshold with respect to the maximum pixel value in the
                               dark frame to flag bad pixels.
        :type dark_threshold: float
        :param flat_threshold: Fractional threshold with respect to the maximum pixel value in the
                               flat field to flag bad pixels.
        :type flat_threshold: float

        :return: None
        """

        super(BadPixelMapModule_IRDIS, self).__init__(name_in)

        if dark_in_tag is None:
            self.m_dark_port = None
        else:
            self.m_dark_port = self.add_input_port(dark_in_tag)

        if flat_in_tag is None:
            self.m_flat_port = None
        else:
            self.m_flat_port = self.add_input_port(flat_in_tag)

        self.m_bp_map_out_port = self.add_output_port(bp_map_out_tag)

        self.m_dark_threshold = dark_threshold
        self.m_flat_threshold = flat_threshold

    def run(self):
        """
        Run method of the module. Collapses a cube of dark frames and flat fields if needed, flags
        bad pixels by comparing the pixel values with the threshold times the maximum value, and
        writes a bad pixel map to the database. For the dark frame, pixel values larger than the
        threshold will be flagged while for the flat frame pixel values smaller than the threshold
        will be flagged.

        :return: None
        """

        if self.m_dark_port is not None:
            dark = self.m_dark_port.get_all()

            if dark.ndim == 3:
                dark = np.mean(dark, axis=0)

            max_dark = np.max(dark)
            print("Threshold dark frame [counts] =", max_dark*self.m_dark_threshold)

            bpmap = np.ones(dark.shape)
            bpmap[np.where(dark > max_dark*self.m_dark_threshold)] = 0

        if self.m_flat_port is not None:
            flat = self.m_flat_port.get_all()

            if flat.ndim == 3:
                flat = np.mean(flat, axis=0)

            max_flat = np.max(flat)
            print("Threshold flat field [counts] =", max_flat*self.m_flat_threshold)

            if self.m_dark_port is None:
                bpmap = np.ones(flat.shape)

            bpmap[np.where(flat < max_flat*self.m_flat_threshold)] = 0
            bpmap[np.where(flat == 1)] = 0     # Extra condition, if master flat comes from sphere DC

        # Mask unimportant part of IRDIS
        rect_apertures = RectangularAperture(positions=[(487,520),(1513,509)],
                                             w=883,
                                             h=996,
                                             theta=0.)

        rect_apertures_mask = (rect_apertures.to_mask(method="center")[0]).to_image((1024, 2048)) +\
                              (rect_apertures.to_mask(method="center")[1]).to_image((1024, 2048))

        bpmap[rect_apertures_mask == 0.] = 1

        if self.m_dark_port is not None and self.m_flat_port is not None:
            if not dark.shape == flat.shape:
                raise ValueError('Dark and flat images should have the same shape.')

        self.m_bp_map_out_port.set_all(bpmap, data_dim=3)

        if self.m_dark_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_dark_port)
        elif self.m_flat_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_flat_port)

        history = f'dark = {self.m_dark_threshold}, flat = {self.m_flat_threshold}'
        self.m_bp_map_out_port.add_history('BadPixelMapModule', history)

        self.m_bp_map_out_port.close_port()


class BadPixelMapModule_NACO(ProcessingModule):
    """
    Module to create a bad pixel map from the dark frames and flat fields.
    """

    def __init__(self,
                 name_in="bad_pixel_map",
                 dark_in_tag="dark",
                 flat_in_tag="flat",
                 bp_map_out_tag="bp_map",
                 dark_threshold=0.2,
                 flat_threshold=0.2):
        """
        Constructor of BadPixelMapModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param dark_in_tag: Tag of the database entry with the dark frames that are read as input.
                            Not read if set to None.
        :type dark_in_tag: str
        :param flat_in_tag: Tag of the database entry with the flat fields that are read as input.
                            Not read if set to None.
        :type flat_in_tag: str
        :param bp_map_out_tag: Tag of the database entry with the bad pixel map that is written as
                               output.
        :type bp_map_out_tag: str
        :param dark_threshold: Fractional threshold with respect to the maximum pixel value in the
                               dark frame to flag bad pixels.
        :type dark_threshold: float
        :param flat_threshold: Fractional threshold with respect to the maximum pixel value in the
                               flat field to flag bad pixels.
        :type flat_threshold: float

        :return: None
        """

        super(BadPixelMapModule_NACO, self).__init__(name_in)

        if dark_in_tag is None:
            self.m_dark_port = None
        else:
            self.m_dark_port = self.add_input_port(dark_in_tag)

        if flat_in_tag is None:
            self.m_flat_port = None
        else:
            self.m_flat_port = self.add_input_port(flat_in_tag)

        self.m_bp_map_out_port = self.add_output_port(bp_map_out_tag)

        self.m_dark_threshold = dark_threshold
        self.m_flat_threshold = flat_threshold

    def run(self):
        """
        Run method of the module. Collapses a cube of dark frames and flat fields if needed, flags
        bad pixels by comparing the pixel values with the threshold times the maximum value, and
        writes a bad pixel map to the database. For the dark frame, pixel values larger than the
        threshold will be flagged while for the flat frame pixel values smaller than the threshold
        will be flagged.

        :return: None
        """

        if self.m_dark_port is not None:
            dark = self.m_dark_port.get_all()

            if dark.ndim == 3:
                dark = np.mean(dark, axis=0)

            max_dark = np.max(dark)
            print("Threshold dark frame [counts] =", max_dark*self.m_dark_threshold)

            bpmap = np.ones(dark.shape)
            bpmap[np.where(dark > max_dark*self.m_dark_threshold)] = 0

        if self.m_flat_port is not None:
            flat = self.m_flat_port.get_all()

            if flat.ndim == 3:
                flat = np.mean(flat, axis=0)

            max_flat = np.max(flat)
            print("Threshold flat field [counts] =", max_flat*self.m_flat_threshold)

            if self.m_dark_port is None:
                bpmap = np.ones(flat.shape)

            bpmap[np.where(flat < max_flat*self.m_flat_threshold)] = 0

        im_size = dark.shape[0]

        # Mask unimportant part of NACO
        rect_apertures = RectangularAperture(positions=(im_size/4+2,im_size/4+2),
                                             w=im_size/2+5,
                                             h=im_size/2+5,
                                             theta=0.)

        rect_apertures_mask = rect_apertures.to_mask(method="center").to_image((im_size, im_size))

        bpmap[rect_apertures_mask == 1.] = 1

        if self.m_dark_port is not None and self.m_flat_port is not None:
            if not dark.shape == flat.shape:
                raise ValueError('Dark and flat images should have the same shape.')

        self.m_bp_map_out_port.set_all(bpmap, data_dim=3)

        if self.m_dark_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_dark_port)
        elif self.m_flat_port is not None:
            self.m_bp_map_out_port.copy_attributes(self.m_flat_port)

        history = f'dark = {self.m_dark_threshold}, flat = {self.m_flat_threshold}'
        self.m_bp_map_out_port.add_history('BadPixelMapModule', history)

        self.m_bp_map_out_port.close_port()



class ReplaxeBadLines_NACO(ProcessingModule):
    """
    Module to correct for bad pixels lines of the NACO detector
    """

    def __init__(self,
                 name_in="replace_bad_lines_NACO",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_corrected"):
        """
        Constructor of ReplaxeBadLines_NACO.

        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output.

        Returns
        -------
        NoneType
            None
        """

        super(ReplaxeBadLines_NACO, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

    def run(self):
        """
        Run module

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_data()
        self.m_image_out_port.del_all_attributes()

        def _replace_pixels(image):

            im_mask = np.copy(image)

            index = np.argwhere(im_mask == np.max(im_mask))

            for _, item in enumerate(index):
                im_mask[item[0], item[1]] = np.nan

            # from matplotlib import pyplot as plt
            # plt.imshow(im_mask, origin="lower")
            # plt.show()

            for _, item in enumerate(index):

                im_mask[item[0], item[1]] = np.nanmean(im_mask[item[0],
                                                               item[1]-2:item[1]+2])

                # im_tmp = im_mask[item[0]-self.m_size:item[0]+self.m_size+1,
                #                  item[1]-self.m_size:item[1]+self.m_size+1]
                #
                # if np.size(np.where(im_tmp != np.nan)[0]) == 0:
                #     im_mask[item[0], item[1]] = image[item[0], item[1]]
                # else:
                #     if self.m_replace == "mean":
                #         im_mask[item[0], item[1]] = np.nanmean(im_tmp)
                #     elif self.m_replace == "median":
                #         im_mask[item[0], item[1]] = np.nanmedian(im_tmp)
                #     elif self.m_replace == "nan":
                #         im_mask[item[0], item[1]] = np.nan

            return im_mask

        self.apply_function_to_images(_replace_pixels,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ReplaxeBadLines_NACO...",
                                      func_args=())

        history = "replace bad lines NACO"
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("ReplaxeBadLines_NACO", history)
        self.m_image_out_port.close_port()
