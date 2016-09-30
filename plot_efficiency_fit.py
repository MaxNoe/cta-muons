import argparse
from astropy import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.linalg import rotation_matrix_2d
from ctapipe.visualization import CameraDisplay
from ctapipe.io.camera import CameraGeometry
from ctapipe.calib.camera.calibrators import (
    calibrate_event, calibration_arguments, calibration_parameters
)
from ctapipe.calib.array.muon.fitting import expected_pixel_light_content

from itertools import chain


def rotate(x, y, angle):
    x, y = np.dot(rotation_matrix_2d(angle), [x, y])
    return x, y


parser = argparse.ArgumentParser(description='Display each event in the file')
parser.add_argument('inputfile')
parser.add_argument('fit_results')
parser.add_argument('outputfile')
parser.add_argument('--num-threads', '-n', dest='n_jobs', type=int, default=-1)


def main():

    calibration_arguments(parser)
    args = parser.parse_args()

    params = calibration_parameters(args)

    log.debug("[file] Reading file")

    source = hessio_event_source(args.inputfile, allowed_tels=[1])
    fit_results = pd.read_hdf(args.fit_results)

    fit_results['radius_degrees'] = np.rad2deg(fit_results['radius'] / 28.0)

    fit_results = fit_results.query(
        'ring_completeness > 0.6'
        ' & time_std < 1.5'
        ' & size > 1000'
        ' & 0.4 < radius_degrees < 1.3'
    )
    print(len(fit_results))

    fit_results.set_index('event_number', inplace=True)
    event = next(source)

    pixel_x = event.meta.pixel_pos[1][0]
    pixel_y = event.meta.pixel_pos[1][1]
    geom = CameraGeometry.guess(pixel_x, pixel_y, event.meta.optical_foclen[1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    disp1 = CameraDisplay(geom, ax=ax1)
    disp1.add_colorbar(ax=disp1.axes)
    disp2 = CameraDisplay(geom, ax=ax2)
    disp2.add_colorbar(ax=disp2.axes)
    circs = [
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=1),
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=1, linestyle=':'),
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=1),
    ]
    for circ in circs:
        disp1.axes.add_artist(circ)

    for disp in (disp2, disp1):
        disp.axes.set_xlim(-1.2, 1.2)
        disp.axes.set_ylim(-1.2, 1.2)

    fig.tight_layout()

    disp1.axes.set_title('Measured Event')
    disp2.axes.set_title('Model according to fit results')

    with PdfPages(args.outputfile) as pdf:
        for event in chain([event], source):

            if event.count not in fit_results.index:
                continue

            print(event.count)

            disp2.axes

            fig.suptitle('Event: {}'.format(event.count))

            fit_result = fit_results.loc[event.count]
            x, y = rotate(
                fit_result.center_x,
                fit_result.center_y,
                geom.pix_rotation,
            )

            for alpha, circ in zip((-1, 0, 1), circs):
                circ.radius = fit_result.radius + alpha * fit_result.sigma
                circ.center = x, y

            calib_event = calibrate_event(event, params)
            mask = calib_event.dl1.tel[1].pe_charge > 10

            linewidth = np.zeros_like(disp1.image)
            color = np.zeros((len(disp1.image), 4))
            color[:, 3] = 0.7
            color[mask, 1] = 1
            linewidth[mask] = 0.75

            disp1.pixels.set_linewidth(linewidth)
            disp1.pixels.set_edgecolors(color)
            disp1.image = calib_event.dl1.tel[1].pe_charge

            disp2.image = fit_result.efficiency * expected_pixel_light_content(
                pixel_x=pixel_x.value,
                pixel_y=pixel_y.value,
                center_x=fit_result.center_x,
                center_y=fit_result.center_y,
                phi_max=fit_result.phi_max,
                cherenkov_angle=fit_result.radius / 28.0,
                impact_parameter=fit_result.impact_parameter,
                sigma_psf=fit_result.sigma,
                pixel_fov=np.deg2rad(0.1),
                pixel_diameter=3.8e-2,
                mirror_radius=11.5,
                focal_length=28,
            )
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
