# -*- coding: utf-8 -*-
from astropy.time import Time
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# from AstrometryTools.pm_analysis_backwards import draworbitangle_back, period, calcpa_parallax_back_pmerror, \
#     hms2decimal, dms2decimal, draworbitsep_back, calcsep_parallax_back_pmerror

__author__ = "Alexander Bohn"

import numpy as np
from matplotlib import pyplot as plt
from uncertainties import ufloat, umath
from astropy.coordinates import SkyCoord
from astropy import units as u
import math
from scipy.interpolate import interp1d
from astropy import constants as c

# def drawpmdiagram_backwards(M,
#                             a,
#                             aerror,
#                             angle,
#                             anerror,
#                             starttime,
#                             timelength,
#                             parallax,
#                             ramotion,
#                             decmotion,
#                             ramotion_error,
#                             decmotion_error,
#                             JD,
#                             ra_0,
#                             dec_0,
#                             previous_data):
#
#    previous_data.pprint(max_lines=-1,max_width=-1)
#
#    t=np.arange(starttime - timelength,starttime,0.1)
#    angle_array = np.linspace(angle,angle,len(t))
#    sep_array = np.linspace(a,a,len(t))
#
#    # PA:
#    fig = plt.figure(1)
#    ax = plt.subplot(111)
#    ax = plt.subplot(211)
#    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#    plt.ylabel("Position angle [deg]", fontsize = 10)
#    # plt.xlabel("Epoch [year]", fontsize = 10)
#    plt.errorbar(starttime, angle, yerr=anerror, fmt='C3o',label=r"SPHERE 2017")
#
#    for i in range(len(previous_data)):
#       plt.errorbar(previous_data["epoch"][i],
#                previous_data["pos_ang"][i],
#                yerr=previous_data["pos_ang_err"][i],
#                fmt="C%is"%(i*3+1),
#                label=previous_data["id"][i])
#
#    t_orb,max = draworbitangle_back(period(M,a,parallax),angle,anerror,starttime,timelength)
#    t_para,pa_para_er,pa_para = calcpa_parallax_back_pmerror(hms2decimal(ra_0,":"),dms2decimal(dec_0,":"),ramotion,decmotion,ramotion_error,decmotion_error,parallax,a,angle,anerror,starttime,timelength,JD)
#    plt.plot(t,angle_array,'C3-.')
#    prop = FontProperties(size=12)
#    ax.legend(loc=0, frameon=True, labelspacing=1, bbox_to_anchor=(.06, 1.),ncol=len(previous_data)+1)
#
#    #SEP:
#    ax = plt.subplot(212)
#    plt.ylabel("Separation [arcsec]", fontsize = 10)
#    plt.xlabel("Epoch [year]", fontsize = 10)
#    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#    plt.errorbar(starttime, a, yerr=aerror, fmt='C3o')
#
#
#    for i in range(len(previous_data)):
#       plt.errorbar(previous_data["epoch"][i],
#                previous_data["sep"][i],
#                yerr=previous_data["sep_err"][i],
#                fmt="C%is"%(i*3+1),
#                label=previous_data["id"][i])
#
#    # plot_linear_trend(55.6562379016204,-0.0261262947515556,t)
#    t_orb,max = draworbitsep_back(M,parallax,a,aerror,starttime,timelength)
#    t_para,sep_para_er,sep_para = calcsep_parallax_back_pmerror(hms2decimal(ra_0,":"),dms2decimal(dec_0,":"),ramotion,decmotion,ramotion_error,decmotion_error,parallax,a,aerror,angle,starttime,timelength,JD)
#    plt.plot(t,sep_array,'C3-.')
#    prop = FontProperties(size=12)
#    # plt.legend(loc=0,markerscale=0.9,numpoints=1,ncol=2,labelspacing=0.5,handletextpad=0.5,prop=prop)
#    # plt.show()
#    return

class ProperMotionDiagram():

    def __init__(self):

        # set parameters
        self.m_epoch = 0.
        self.m_test_dates = np.array([])
        self.m_test_delta_ra = np.array([])
        self.m_test_delta_dec = np.array([])
        self.m_test_delta_ra_err = np.array([])
        self.m_test_delta_dec_err = np.array([])

        self.m_test_markers = np.array([])
        self.m_test_colors = np.array([])
        self.m_test_alphas = np.array([])
        self.m_test_labels = np.array([])

    def add_target(self,
                   name,
                   parallax,
                   parallax_err,
                   ra,
                   dec,
                   epoch,
                   pm_ra,
                   pm_dec,
                   pm_ra_err,
                   pm_dec_err,
                   M_star=1.,
                   M_comp=1.):

        self.m_target_name = name
        self.m_M_star = M_star
        self.m_M_comp = M_comp
        self.m_parallax = parallax
        self.m_parallax_err = parallax_err
        self.m_ra = ra
        self.m_dec = dec
        self.m_epoch = epoch
        self.m_pm_ra = pm_ra
        self.m_pm_dec = pm_dec
        self.m_pm_ra_err = pm_ra_err
        self.m_pm_dec_err = pm_dec_err

    def polar_to_cartesian(self,
                           sep,
                           ang,
                           sep_err,
                           ang_err):
        """
        Function to convert polar coordinates (sep, ang) to cartesian coordinates (delta_ra, delta_dec).

        :param sep: Separation (mas).
        :type sep: float
        :param ang: Position angle (deg), measured counterclockwise with respect to the
                    positive y-axis.
        :type ang: float

        :return: Cartesian coordinates (delta_ra, delta_dec).
        :rtype: float, float
        """

        delta_ra = ufloat(sep, sep_err) * umath.cos(umath.radians(ufloat(ang, ang_err) + 90.))
        delta_dec = ufloat(sep, sep_err) * umath.sin(umath.radians(ufloat(ang, ang_err) + 90.))

        return -1 * delta_ra.nominal_value, delta_dec.nominal_value, delta_ra.std_dev, delta_dec.std_dev

    def cartesian_to_polar(self,
                           delta_ra,
                           delta_dec,
                           delta_ra_err,
                           delta_dec_err):
        """
        Function to convert cartesian coordinates (delta_ra, delta_dec) to polar coordinates (sep, ang).

        :param delta_ra: Offset in RA (mas).
        :type sep: float
        :param delta_dec: Offset in Dec (mas)
        :type ang: float
        :param delta_ra_err: Uncertainty in RA (mas).
        :type sep: float
        :param delta_dec_err: Uncertainty in Dec (mas)
        :type ang: float

        :return: Polar coordinates (sep, pa, sep_err, pa_err).
        :rtype: float, float, float, float
        """

        sep = umath.sqrt(ufloat(delta_ra, delta_ra_err) ** 2 + ufloat(delta_dec, delta_dec_err) ** 2)
        pa = umath.atan2(ufloat(-delta_ra, delta_ra_err), ufloat(-delta_dec, delta_dec_err)) / np.pi * 180. + 180

        return sep.nominal_value, pa.nominal_value, sep.std_dev, pa.std_dev

    def add_reference_epoch(self,
                            date,
                            separation,
                            position_angle,
                            separation_err,
                            position_angle_err,
                            marker="o",
                            color="C1",
                            label=None):

        self.m_ref_date = date
        self.m_ref_delta_ra, self.m_ref_delta_dec, self.m_ref_delta_ra_err, self.m_ref_delta_dec_err = \
            self.polar_to_cartesian(sep=separation,
                                    ang=position_angle,
                                    sep_err=separation_err,
                                    ang_err=position_angle_err)
        self.m_ref_marker = marker
        self.m_ref_color = color
        self.m_ref_label = label

    def add_test_epochs(self,
                        date,
                        separation,
                        position_angle,
                        separation_err,
                        position_angle_err,
                        marker="o",
                        color="C1",
                        alpha=None,
                        label=None):

        self.m_test_dates = np.append(self.m_test_dates,date)

        for i, _ in enumerate(date):

            tmp_position = self.polar_to_cartesian(sep=separation[i],
                                                   ang=position_angle[i],
                                                   sep_err=separation_err[i],
                                                   ang_err=position_angle_err[i])
            
            self.m_test_delta_ra = np.append(self.m_test_delta_ra,tmp_position[0])
            self.m_test_delta_dec = np.append(self.m_test_delta_dec,tmp_position[1])
            self.m_test_delta_ra_err = np.append(self.m_test_delta_ra_err,tmp_position[2]) 
            self.m_test_delta_dec_err = np.append(self.m_test_delta_dec_err,tmp_position[3])

            self.m_test_markers = np.append(self.m_test_markers,marker)
            self.m_test_colors = np.append(self.m_test_colors,color)
            self.m_test_alphas = np.append(self.m_test_alphas,alpha)
            if i == 0:
                self.m_test_labels = np.append(self.m_test_labels,label)
            else:
                if label in self.m_test_labels:
                    self.m_test_labels = np.append(self.m_test_labels,None)
                else:
                    self.m_test_labels = np.append(self.m_test_labels,label)

    def get_parallax_correction(self,
                                date):

        def period360(x):
            if (x < 0):
                x = - x
                y = x % 360
                x = 360 - y
            if (x > 360):
                x = x % 360
            return x

        # mean anomaly - muss zwischen 0 und 360 liegen:

        def mean_anom(JD):
            g = 357.528 + 0.9856003 * (JD - 2451545.0)
            g = period360(g)

            return g

        # mean Longitude of sun - muss zwischen 0 und 360 liegen:

        def mean_Long(JD):
            L = 280.460 + 0.9856474 * (JD - 2451545.0)
            L = period360(L)
            return L

        # ecliptic Longitude:

        def ecl_Long(JD):
            L = mean_Long(JD)
            g = math.radians(mean_anom(JD))
            lam = L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)
            return lam

        # distance earth from sun in AU:

        def sun_dist(JD):
            g = math.radians(mean_anom(JD))
            R = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
            return R

        # Schiefe der Ekliptik

        def eclip(JD):
            eps = 23.439 - 0.0000004 * (JD - 2451545.0)
            return eps

        # --- Sun coordinates

        def x_sun(JD):

            R = sun_dist(JD)
            lam = math.radians(ecl_Long(JD))
            x = - R * math.cos(lam)

            return x

        def y_sun(JD):

            R = sun_dist(JD)
            lam = math.radians(ecl_Long(JD))
            eps = math.radians(eclip(JD))
            y = - R * math.cos(eps) * math.sin(lam)

            return y

        def z_sun(JD):

            R = sun_dist(JD)
            lam = ecl_Long(JD)
            eps = eclip(JD)
            lam = lam / 180. * math.pi
            eps = eps / 180. * math.pi
            z = - R * math.sin(eps) * math.sin(lam)

            return z

        parallax_correction_ra = self.m_parallax * (x_sun(date) * math.sin(math.radians(self.m_ra)) -
                                                    y_sun(date) * math.cos(math.radians(self.m_ra)))

        parallax_correction_dec = self.m_parallax * (x_sun(date) * math.cos(math.radians(self.m_ra)) * math.sin(math.radians(self.m_dec)) +
                                                     y_sun(date) * math.sin(math.radians(self.m_ra)) * math.sin(math.radians(self.m_dec)) -
                                                     z_sun(date) * math.cos(math.radians(self.m_dec)))

        return parallax_correction_ra, parallax_correction_dec


    def get_period(self,
                   a):

        # Kepler's third law
        period = np.sqrt(a**3/(c.G*((self.m_M_star + self.m_M_comp) * u.M_sun/(4*np.pi**2))))

        return period



    def plot_ppm_diagram(self,
                         plot_orbital_motion=False,
                         xlim=None,
                         ylim=None,
                         title=None,
                         show=True,
                         path_save=None):
        #
        # plt.style.use('dark_background')
        fig, ax = plt.subplots(1,1, figsize = (8,8))

        # plot reference epoch
        # print(self.m_ref_delta_ra_err, self.m_ref_delta_dec_err)
        ax.errorbar(x=self.m_ref_delta_ra,
                    y=self.m_ref_delta_dec,
                    xerr=self.m_ref_delta_ra_err,
                    yerr=self.m_ref_delta_dec_err,
                    ecolor="grey",
                    elinewidth=1.,
                    capsize=1.,
                    capthick=1.,
                    linestyle='',
                    marker=self.m_ref_marker,
                    markerfacecolor=self.m_ref_color,
                    markeredgecolor=self.m_ref_color,
                    markersize=8,
                    zorder=2,
                    label=self.m_ref_label)

        # plot test epochs
        for i, _ in enumerate(self.m_test_dates):
            # print(self.m_test_delta_ra_err[i], self.m_test_delta_dec_err[i])
            ax.errorbar(x=self.m_test_delta_ra[i],
                        y=self.m_test_delta_dec[i],
                        xerr=self.m_test_delta_ra_err[i],
                        yerr=self.m_test_delta_dec_err[i],
                        ecolor="grey",
                        elinewidth=1.,
                        capsize=1.,
                        capthick=1.,
                        linestyle='',
                        marker=self.m_test_markers[i],
                        markerfacecolor=self.m_test_colors[i],
                        markeredgecolor='none',
                        markersize=8,
                        alpha=self.m_test_alphas[i],
                        zorder=2,
                        label=self.m_test_labels[i])

        # get background track
        base_coordinate = SkyCoord(ra=self.m_ra * u.deg,
                                      dec=self.m_dec * u.deg,
                                      distance=1000./self.m_parallax*u.pc,
                                      frame="icrs",
                                      obstime=self.m_epoch,
                                      equinox="J2015.5",
                                      representation_type="spherical",
                                      pm_ra_cosdec=self.m_pm_ra*u.mas/u.yr * math.cos(math.radians(self.m_dec)),
                                      pm_dec=self.m_pm_dec*u.mas/u.yr)

        # combine ref date with test dates
        all_dates = np.append(self.m_test_dates,self.m_ref_date)

        index_max_date = np.argmax(Time(all_dates).jd1)
        index_min_date = np.argmin(Time(all_dates).jd1)

        n = 1000

        ref_epoch_coordinate = base_coordinate.apply_space_motion(new_obstime=Time(self.m_ref_date))
        time_interval_coordinates = base_coordinate.apply_space_motion(new_obstime=Time([all_dates[index_min_date], all_dates[index_max_date]]))

        parallax_correction_ra = np.array([])
        parallax_correction_dec = np.array([])

        ref_parallax_correction_ra, ref_parallax_correction_dec = \
            self.get_parallax_correction(Time(self.m_ref_date).jd1)


        for tmp_time in np.linspace(Time(all_dates[index_min_date]).jd1, Time(all_dates[index_max_date]).jd1,num=n):

            plx_corr_ra, plx_corr_dec = self.get_parallax_correction(tmp_time)

            parallax_correction_ra = np.append(parallax_correction_ra,
                                               plx_corr_ra)
            parallax_correction_dec = np.append(parallax_correction_dec,
                                                plx_corr_dec)

        # Set first offset to 0
        parallax_correction_ra -= ref_parallax_correction_ra
        parallax_correction_dec -= ref_parallax_correction_dec

        ra_bg_track = -1 * np.linspace(time_interval_coordinates.ra[0].to(u.mas)-ref_epoch_coordinate.ra.to(u.mas), time_interval_coordinates.ra[1].to(u.mas)-ref_epoch_coordinate.ra.to(u.mas),num=n) + self.m_ref_delta_ra * u.mas - parallax_correction_ra*u.mas
        dec_bg_track = -1 * np.linspace(time_interval_coordinates.dec[0].to(u.mas)-ref_epoch_coordinate.dec.to(u.mas), time_interval_coordinates.dec[1].to(u.mas)-ref_epoch_coordinate.dec.to(u.mas),num=n) + self.m_ref_delta_dec * u.mas - parallax_correction_dec*u.mas

        # get static background positions
        ra_bg_object = interp1d(np.linspace(Time(all_dates[index_min_date]).jd1, Time(all_dates[index_max_date]).jd1,num=n),
                                ra_bg_track)(Time(self.m_test_dates).jd1)
        dec_bg_object = interp1d(np.linspace(Time(all_dates[index_min_date]).jd1, Time(all_dates[index_max_date]).jd1,num=n),
                                 dec_bg_track)(Time(self.m_test_dates).jd1)

#         print(self.m_test_delta_ra, self.m_test_delta_ra_err)
#         print(self.m_test_delta_dec, self.m_test_delta_dec_err)
#         print(ra_bg_object, dec_bg_object)
#         print(self.m_ref_delta_ra_err, self.m_ref_delta_dec_err)

        # plot static background positions
        for i, _ in enumerate(self.m_test_dates):
            ax.errorbar(x=ra_bg_object[i],
                        y=dec_bg_object[i],
                        xerr=self.m_ref_delta_ra_err,
                        yerr=self.m_ref_delta_dec_err,
                        ecolor="grey",
                        elinewidth=1.,
                        capsize=1.,
                        capthick=1.,
                        linestyle='',
                        marker=self.m_test_markers[i],
                        markerfacecolor='w',
                        markeredgecolor='k',
                        markersize=8,
                        zorder=1)

        ax.plot(ra_bg_track,
                dec_bg_track,
                color="C0",
                linestyle="--",
                zorder=0)

        if plot_orbital_motion:
            period = self.get_period(a=1./self.m_parallax*np.sqrt(self.m_ref_delta_ra**2+self.m_ref_delta_dec**2)*u.au)
#             print(period.to(u.d))
#             print(2*np.pi*1./self.m_parallax*np.sqrt(self.m_ref_delta_ra**2+self.m_ref_delta_dec**2)*1./period.to(u.yr).value)
            # for tmp_time in self.m_test_dates:
            # TODO


        ax.tick_params(axis="both",
                       labelsize=12
                       )

        ax.set_xlabel(r"$\Delta$RA [mas]", fontsize=12)
        ax.set_ylabel(r"$\Delta$Dec [mas]", fontsize=12)
        ax.invert_xaxis()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=0, frameon=False)
        ax.legend(loc=0, frameon=True)
        if title != None:
            ax.set_title(title, fontsize=20)
        if xlim != None:
            ax.set_xlim(xlim)
        if ylim != None:
            ax.set_ylim(ylim)

        # evaluate at how many sigma the object is bound
        err_1 = (self.m_test_delta_ra_err[0]+self.m_test_delta_dec_err[0])/2
        err_2 = (self.m_ref_delta_ra_err+self.m_ref_delta_dec_err)/2
        dist = (np.sqrt((self.m_test_delta_ra[0]-ra_bg_object[0])**2 +
                       (self.m_test_delta_dec[0]-dec_bg_object[0])**2))

        # print((dist)/(err_1+err_2))

        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(40))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        # tick parameters
        ax.tick_params(axis="both",
                       reset=False,
                       which="major",
                       direction="in",
                       length=6.,
                       width=2.,
                       labelsize=12,
                       colors="k",
                       labelcolor="k",
                       top=True,
                       bottom=True,
                       right=True,
                       left=True,
                       labelbottom=True,
                       labeltop=False,
                       labelright=False,
                       labelleft=True)

        # tick parameters
        ax.tick_params(axis="both",
                       reset=False,
                       which="minor",
                       direction="in",
                       length=3.,
                       width=2,
                       colors="k",
                       labelcolor="black",
                       top=True,
                       bottom=True,
                       right=True,
                       left=True,
                       labelbottom=True,
                       labeltop=False,
                       labelright=False,
                       labelleft=True)

        if show==True:
            plt.show()

        if path_save != None:
            fig.savefig(path_save,bbox_inches='tight', transparent=True, pad_inches=0)
        else:
            return fig, ax

        plt.close(fig)
