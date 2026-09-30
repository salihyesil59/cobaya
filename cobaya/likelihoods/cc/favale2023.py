import os

from cobaya.likelihoods.base_classes import CC
from cobaya.log import LoggedError

_covariance_choices = {
    "file": "the covariance matrix given in `covmat_file` (e.g. the authors' one)",
    "moresco2020": "the tabulated errors plus the IMF and SPS systematics of Moresco et "
    "al. (2020), fully correlated across redshift (needs `cobaya-install "
    "cc.moresco2020`)",
    "uncorrelated": "the tabulated errors only, uncorrelated",
}


class favale2023(CC):
    r"""
    Compilation of 32 cosmic chronometers :math:`H(z)` measurements of
    \cite{Favale:2023}, Table 1.

    The covariance matrix used in that work is not public, and it cannot be recovered
    unambiguously from the paper, so it has to be chosen with the ``covariance`` option.
    The paper builds it with the method of \cite{Moresco:2020}, which corresponds to
    ``covariance: moresco2020``. But for the 15 measurements of Moresco et al. the
    tabulated errors seem to include the SPS-model systematic already (they are close
    to the quadrature sum of the statistical error and that systematic), in which case
    that option counts it twice for them. Preferably, use ``covariance: file`` with the
    covariance matrix of the authors.
    """

    covariance: str | None

    def initialize(self):
        if self._is_default("data_file"):
            self.data_file = os.path.join(self.get_class_path(), self.data_file)
        if self.covariance not in _covariance_choices:
            raise LoggedError(
                self.log,
                "The covariance of this compilation is not public, so it must be chosen "
                "with the 'covariance' option (see the documentation of this likelihood "
                "for the caveats): %s.",
                "; ".join(f"'{k}': {v}" for k, v in _covariance_choices.items()),
            )
        if self.covariance == "file":
            if not self.covmat_file:
                raise LoggedError(
                    self.log, "'covariance: file' requires a 'covmat_file'."
                )
            self.systematics = []
        elif self.covariance == "uncorrelated":
            self.covmat_file = None
            self.systematics = []
        else:  # moresco2020
            self.covmat_file = None
            self.systematics = ["IMF", "mod_ooo"]
            if self._is_default("systematics_file"):
                from cobaya.likelihoods.cc.moresco2020 import _commit, moresco2020

                folder = moresco2020.get_path(self.packages_path or "")
                self.systematics_file = os.path.join(
                    folder, f"CCcovariance-{_commit}", "data", "data_MM20.dat"
                )
                if not os.path.exists(self.systematics_file):
                    raise LoggedError(
                        self.log,
                        "'covariance: moresco2020' needs the systematics of Moresco et "
                        "al. (2020): run `cobaya-install cc.moresco2020`.",
                    )
        super().initialize()
