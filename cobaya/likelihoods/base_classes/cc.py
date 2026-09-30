r"""
.. module:: cc

:Synopsis: Cosmic chronometers likelihood class
:Author: Salih Yesil

Cosmic chronometers (CC) measure the Hubble rate :math:`H(z)` from the differential age
evolution of passively evolving galaxies. This is a Gaussian likelihood for a set of such
:math:`H(z)` measurements, including the systematic uncertainties of the method with the
full covariance matrix of `Moresco et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2020ApJ...898...82M/abstract>`_:

.. math::

   C_{ij} = \sigma_i^2\,\delta_{ij} + \sum_k H_i f_k(z_i)\, H_j f_k(z_j)\,,

where :math:`\sigma_i` are the tabulated (uncorrelated) errors of the measurements and
:math:`f_k(z)` are the fractional systematic errors of the components :math:`k` listed
in ``systematics`` (by default the initial mass function and the stellar population
synthesis model), which are fully correlated across redshift. The fractional errors are
linearly interpolated in redshift from the table of Moresco et al. (2020), and held at
the values at the ends of the table beyond it.

This is the construction of the public `CCcovariance
<https://gitlab.com/mmoresco/CCcovariance>`_ repository by M. Moresco, from which the
data files of ``cc.moresco2020`` are taken unmodified.

Using a different set of measurements
-------------------------------------

To use a different set of :math:`H(z)` measurements, create a class inheriting from
:class:`cc.CC`, with a ``.yaml`` file setting ``data_file`` to a text file whose first
three columns are :math:`z`, :math:`H(z)` and its uncorrelated error, in km/s/Mpc
(comma- or space-separated). Relative paths are understood with respect to the folder
of the likelihood class. Set ``systematics: []`` to use only the uncorrelated errors.
"""

import os

import numpy as np

from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError


class CC(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "CC"

    # variables from yaml
    data_file: str
    systematics_file: str
    systematics: list[str]

    def initialize(self):
        self.z, self.H_data, sigma = self._load_table(self.data_file, (0, 1, 2)).T
        self.cov = np.diag(sigma**2)
        if self.systematics:
            with open(self._path(self.systematics_file)) as f:
                columns = f.readline().lstrip("#").split()
            table = self._load_table(self.systematics_file)
            for name in self.systematics:
                if name not in columns[1:]:
                    raise LoggedError(
                        self.log,
                        "Unknown systematic '%s'. Available ones are %r.",
                        name,
                        columns[1:],
                    )
                fractional = np.interp(
                    self.z, table[:, 0], table[:, columns.index(name)] / 100
                )
                self.cov += np.outer(self.H_data * fractional, self.H_data * fractional)
        self.invcov = np.linalg.inv(self.cov)

    def _path(self, filename):
        return os.path.join(self.get_class_path(), filename)

    def _load_table(self, filename, usecols=None):
        path = self._path(filename)
        try:
            with open(path) as f:
                first_data_line = next(line for line in f if not line.startswith("#"))
            return np.atleast_2d(
                np.loadtxt(
                    path,
                    comments="#",
                    delimiter="," if "," in first_data_line else None,
                    usecols=usecols,
                )
            )
        except (OSError, StopIteration, ValueError) as excpt:
            raise LoggedError(
                self.log, "Could not read data file '%s': %s", path, excpt
            ) from excpt

    def get_requirements(self):
        return {"Hubble": {"z": self.z}}

    def logp(self, **params_values):
        H_theory = self.provider.get_Hubble(self.z, units="km/s/Mpc")
        residual = self.H_data - H_theory
        return -0.5 * residual @ self.invcov @ residual
