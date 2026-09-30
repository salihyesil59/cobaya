"""
Tests the cosmic chronometers likelihood against the published fit of the covariance
recipe it implements, and checks that it is consistent across Boltzmann codes.
"""

import numpy as np
import pytest

from cobaya.model import get_model
from cobaya.theory import Theory

from .common import process_packages_path
from .conftest import install_test_wrapper


class FlatLCDMHubble(Theory):
    """Flat LCDM H(z) without radiation, as used in the reference fit."""

    params = {"H0": None, "Omega_m": None}

    def get_can_provide(self):
        return ["Hubble"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        state["H0"] = params_values_dict["H0"]
        state["Omega_m"] = params_values_dict["Omega_m"]

    def get_Hubble(self, z, units="km/s/Mpc"):
        Omega_m = self.current_state["Omega_m"]
        z = np.asarray(z)
        return self.current_state["H0"] * np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)


def _H0_quantiles(systematics):
    """
    Integrates the posterior of the reference fit (flat priors 50 < H0 < 100,
    0.01 < Omega_m < 0.99) on a grid, returning the median of H0 and the distances to
    the 16% and 84% quantiles.
    """
    model = get_model(
        {
            "likelihood": {"cc.moresco2020": {"systematics": systematics}},
            "theory": {"hubble": FlatLCDMHubble},
            "params": {
                "H0": {"prior": {"min": 50, "max": 100}},
                "Omega_m": {"prior": {"min": 0.01, "max": 0.99}},
            },
        }
    )
    H0s = np.linspace(50, 100, 201)
    Omega_ms = np.linspace(0.01, 0.99, 99)
    loglikes = np.array(
        [[model.loglike([H0, Om], return_derived=False) for Om in Omega_ms] for H0 in H0s]
    )
    marginal = np.exp(loglikes - loglikes.max()).sum(axis=1)
    # cumulative mass up to each grid point (half of its own bin)
    cdf = (np.cumsum(marginal) - marginal / 2) / marginal.sum()
    low, median, high = np.interp([0.16, 0.5, 0.84], cdf, H0s)
    return median, high - median, median - low


def test_cc_moresco2020_reproduces_published_fit():
    # Flat LCDM fit to the same data, with and without the systematics, from the
    # CC_fit.ipynb notebook of https://gitlab.com/mmoresco/CCcovariance, where the
    # data and systematics files come from. That is an emcee run, so it carries
    # sampling noise of a few hundredths of km/s/Mpc.
    assert np.allclose(
        _H0_quantiles(["IMF", "mod_ooo"]), [65.995, 5.545, 5.591], atol=0.1
    )
    assert np.allclose(_H0_quantiles([]), [66.171, 3.770, 3.956], atol=0.1)


def test_cc_moresco2020_boltzmann(packages_path, skip_not_installed):
    # Same chi2 from CAMB and CLASS at a Planck-like cosmology, and equal to the one
    # computed directly from their H(z) with the covariance recipe
    data = np.genfromtxt(
        cc_data_path("HzTable_MM_BC03.dat"), delimiter=",", usecols=(0, 1, 2)
    )
    z, H, sigma = data.T
    systematics = np.loadtxt(cc_data_path("data_MM20.dat"))
    cov = np.diag(sigma**2)
    for column in (1, 4):  # IMF, mod_ooo
        error = H * np.interp(z, systematics[:, 0], systematics[:, column]) / 100
        cov += np.outer(error, error)
    params = {
        "camb": {"H0": 67.4, "ombh2": 0.0224, "omch2": 0.12, "As": 2.1e-9, "ns": 0.965},
        "classy": {
            "H0": 67.4,
            "omega_b": 0.0224,
            "omega_cdm": 0.12,
            "A_s": 2.1e-9,
            "n_s": 0.965,
            # same neutrino content as CAMB's default: one massive, 0.06 eV
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
        },
    }
    chi2s = []
    for theory in ["camb", "classy"]:
        info = {
            "likelihood": {"cc.moresco2020": None},
            "theory": {theory: None},
            "params": params[theory],
            "packages_path": process_packages_path(packages_path),
        }
        model = install_test_wrapper(skip_not_installed, get_model, info)
        chi2 = -2 * model.loglike({}, return_derived=False)
        residual = H - model.provider.get_Hubble(z)
        assert chi2 == pytest.approx(residual @ np.linalg.solve(cov, residual))
        chi2s.append(chi2)
    assert chi2s[0] == pytest.approx(chi2s[1], rel=1e-4)
    assert chi2s[0] < 2 * len(z)


def cc_data_path(filename):
    import os

    import cobaya.likelihoods.cc

    return os.path.join(os.path.dirname(cobaya.likelihoods.cc.__file__), "data", filename)
