import argparse
from astropy import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.linalg import rotation_matrix_2d
from ctapipe.visualization import CameraDisplay
from ctapipe.io.camera import CameraGeometry
from ctapipe.calib.camera.calibrators import (
    calibrate_event, calibration_arguments, calibration_parameters
)

from itertools import chain


def rotate(x, y, angle):
    x, y = np.dot(rotation_matrix_2d(angle), [x, y])
    return x, y


parser = argparse.ArgumentParser(description='Display each event in the file')
parser.add_argument('inputfile')
parser.add_argument('fit_results')
parser.add_argument('--num-threads', '-n', dest='n_jobs', type=int, default=-1)


def main():

    calibration_arguments(parser)
    args = parser.parse_args()

    params = calibration_parameters(args)

    log.debug("[file] Reading file")

    source = hessio_event_source(args.inputfile, allowed_tels=[1])
    fit_results = pd.read_hdf(args.fit_results)

    fit_results['radius_degrees'] = np.rad2deg(fit_results['r'] / 28.0)

    fit_results = fit_results.query(
        'ring_completeness > 0.5 & time_std < 1.5 & size > 600 & 0.4 < radius_degrees < 1.3'
    )

    print(fit_results.head())
    fit_results.set_index('event_number', inplace=True)
    event = next(source)

    pixel_x = event.meta.pixel_pos[1][0]
    pixel_y = event.meta.pixel_pos[1][1]
    geom = CameraGeometry.guess(pixel_x, pixel_y, event.meta.optical_foclen[1])

    disp = CameraDisplay(geom)
    disp.add_colorbar()
    circs = [
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=2),
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=2, linestyle=':'),
        plt.Circle((0, 0), 0, fill=False, edgecolor='c', lw=2),
    ]
    for circ in circs:
        disp.axes.add_artist(circ)
    disp.show()

    for event in chain([event], source):
        print(event.count)

        if event.count not in fit_results.index:
            continue

        print(event.count)

        fit_result = fit_results.loc[event.count]
        print(fit_result)
        x, y = rotate(fit_result.x, fit_result.y, geom.pix_rotation)

        for alpha, circ in zip((-1, 0, 1), circs):
            circ.radius = fit_result.r + alpha * fit_result.sigma
            circ.center = x, y

        calib_event = calibrate_event(event, params)
        disp.image = calib_event.dl1.tel[1].pe_charge

        input('Press enter to continue')


if __name__ == '__main__':
    main()
