import argparse
from astropy import log
from ctapipe.io.hessio import hessio_event_source
import numpy as np
from ctapipe.utils.linalg import rotation_matrix_2d
import pandas as pd
from joblib import Parallel, delayed

from ctapipe.calib.camera.calibrators import (
    calibration_parameters,
    calibrate_event,
    calibration_arguments,
)
from ctapipe.calib.array.muon import (
    psf_likelihood_fit,
    impact_parameter_fit,
    mean_squared_error,
    photon_ratio_inside_ring,
    ring_completeness,
)


def rotate(x, y, angle):
    x, y = np.dot(rotation_matrix_2d(angle), [x, y])
    return x, y


def fit_event(event, pixel_x, pixel_y, params):
    pixel_x = pixel_x.value
    pixel_y = pixel_y.value

    result = {'event_number': event.count}

    event = calibrate_event(event, params)
    photons = event.dl1.tel[1].pe_charge
    time = event.dl1.tel[1].peakpos

    mask = photons > 10
    result['size'] = np.sum(photons[mask])
    result['num_pixel'] = np.sum(mask)

    if result['num_pixel'] < 5:
        return None

    if result['size'] < 200:
        return None

    result['time_std'] = np.std(time[mask])

    r, x, y, sigma = psf_likelihood_fit(
        pixel_x[mask],
        pixel_y[mask],
        photons[mask],
    )
    result.update(r=r, x=x, y=y, sigma=sigma)

    result['ratio_inside'] = photon_ratio_inside_ring(
        pixel_x[mask],
        pixel_y[mask],
        photons[mask],
        center_x=x,
        center_y=y,
        radius=r,
        width=sigma,
    )

    result['mean_squared_error'] = mean_squared_error(
        pixel_x[mask],
        pixel_y[mask],
        photons[mask],
        center_x=x,
        center_y=y,
        radius=r,
    )

    result['ring_completeness'] = ring_completeness(
        pixel_x[mask],
        pixel_y[mask],
        photons[mask],
        center_x=x,
        center_y=y,
        radius=r,
    )

    result['impact_parameter'], result['phi_max'] = impact_parameter_fit(
        pixel_x[mask],
        pixel_y[mask],
        photons[mask],
        center_x=x,
        center_y=y,
        radius=r,
        telescope_radius=11.5,
    )

    return result

parser = argparse.ArgumentParser(description='Display each event in the file')
parser.add_argument('inputfile')
parser.add_argument('--num-threads', '-n', dest='n_jobs', type=int, default=-1)


def main():

    calibration_arguments(parser)
    args = parser.parse_args()

    params = calibration_parameters(args)

    log.debug("[file] Reading file")

    source = hessio_event_source(args.inputfile)
    event = next(source)

    pixel_x = event.meta.pixel_pos[1][0]
    pixel_y = event.meta.pixel_pos[1][1]

    with Parallel(args.n_jobs, verbose=5) as pool:

        result = pool(
            delayed(fit_event)(event, pixel_x, pixel_y, params)
            for event in source
        )

    df = pd.DataFrame(list(filter(lambda x: x is not None, result)))
    df.to_hdf('fit_results.hdf5', 'data')


if __name__ == '__main__':
    main()
