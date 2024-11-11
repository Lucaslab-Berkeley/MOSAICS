import numpy as np

from mosaics.data_structures.utils import parse_single_ctffind5_result
from mosaics.utils import _calculate_pixel_frequency_angle
from mosaics.utils import _calculate_pixel_spatial_frequency


class ContrastTransferFunction:
    r"""Custom class for defining parameters of the contrast transfer function
    (CTF). Note that some of the attributes of the class have associated units,
    but these units are not explicitly enforced.

    The implemented contrast transfer function follows the definition in
    https://doi.org/10.1016/j.jsb.2015.08.008 can be expressed as:

    .. math::
        \begin{equation}
            \text{CTF}(\lambda, \mathbf{g}, \Delta f, C_s, \Delta \phi, w_2) =
            -\sin\left(
                    \chi(\lambda, \mathbf{g}, C_s, \Delta f, \Delta \phi, w_2)
                \right)
        \end{equation}

    where $\lambda$ is the relativistic electron wavelength, \mathbf{g} is the
    spatial frequency, $\Delta f$ is the defocus, $C_s$ is the spherical
    aberration,
    $\Delta \phi$
    is an additional phase shift term, and $w_2$ is the amplitude contrast
    ratio.

    The function $\xi$ is defined as:

    .. math::
        \begin{equation}
        \xi(\lambda, \mathbf{g}, C_s, \Delta f, \Delta \phi, w_2) =
            \pi \lambda \lvert \mathbf{g} \rvert^2 \left[
            \Delta f - \dfrac{1}{2} \lambda^2 C_s \lvert \mathbf{g} \rvert^2
            \right] + \Delta \phi + \arctan\left(w_2 / \sqrt{1 - w_2^2}\right)
        \end{equation}

    There is an additional envelope function that accounts for the overall
    decay of spatial frequency information parameterized here as:

    .. math::
        \begin{equation}
            E = \exp\left(-\dfrac{B}{4}\mathbf{s}^2\right)
        \end{equation}

    where :math:`B` is the B-factor in units of Angstroms^2.

    Attributes:
    -----------

        voltage (float): The voltage of the electron beam in kV.
        wavelength (float): The wavelength of the electron beam in Angstroms.
        spherical_aberration (float): The spherical aberration of the
            microscope in mm.
        z_1 (float): The defocus value along the first axis in Angstroms.
        z_2 (float): The defocus value along the second axis in Angstroms.
        astigmatism_azimuth (float): The astigmatism angle in radians.
        amplitude_contrast_ratio (float): The amplitude contrast ratio.
        B_factor (float): The B-factor in Angstroms^2.
        pixel_size (float): The size of the pixels in Angstroms.

    Methods:
    --------

        compute_ctf(shape: tuple) -> np.ndarray: Compute the contrast transfer
            function with the held CTF parameters for a grid of the given
            shape.
    """

    voltage: float  # in kV
    spherical_aberration: float  # in mm
    defocus_1: float  # in Angstroms
    defocus_2: float  # in Angstroms
    astigmatism_azimuth: float  # in radians
    amplitude_contrast_ratio: float
    B_factor: float  # in Angstroms^2
    pixel_size: float  # in Angstroms, assume square pixels

    @classmethod
    def from_ctffind5_output(
        cls,
        ctffind5_output: str,
        voltage: float,
        spherical_aberration: float,
        amplitude_contrast_ratio: float,
        pixel_size: float,
        B_factor: float = 0.0,
    ) -> "ContrastTransferFunction":
        """Parse a CTFFIND5 fit output file and create a
        ContrastTransferFunction object from the fit parameters.
        """
        fit_params = parse_single_ctffind5_result(ctffind5_output)

        return cls(
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            defocus_1=fit_params["ctffind5.defocus_1"],
            defocus_2=fit_params["ctffind5.defocus_2"],
            astigmatism_azimuth=fit_params["ctffind5.astigmatism_azimuth"],
            amplitude_contrast_ratio=amplitude_contrast_ratio,
            B_factor=B_factor,
            pixel_size=pixel_size,
        )

    def __init__(
        self,
        voltage,
        spherical_aberration,
        defocus_1,
        defocus_2,
        astigmatism_azimuth,
        amplitude_contrast_ratio,
        B_factor,
        pixel_size,
    ):
        self.pixel_size = pixel_size
        self.voltage = voltage
        self.wavelength = self._wavelength_from_voltage(self.voltage)

        self.spherical_aberration = (
            spherical_aberration * 1e7 / pixel_size
        )  # convert to Angstroms, relative to pixel size
        self.defocus_1 = defocus_1
        self.defocus_2 = defocus_2
        self.astigmatism_azimuth = astigmatism_azimuth
        self.amplitude_contrast_ratio = amplitude_contrast_ratio
        self.B_factor = B_factor

    def _wavelength_from_voltage(self, voltage: float) -> float:
        r"""Convert from electron voltage to wavelength accounting for
        relativistic effects. Calculated according to

        .. math::
            \begin{equation}
                \lambda = \dfrac{h}{\sqrt{2m_e e V}}
            \end{equation}

        with :math:`h` as Planck's constant, :math:`m_e` as the electron mass,
        :math:`e` as the elementary charge, and :math:`V` as the voltage. The
        relativistic effects are accounted for by an additional factor of

        .. math::
            \begin{equation}
                \frac{1}{\sqrt{1 + \frac{e\cdot V}{2\cdot m\cdot c^2}}}
            \end{equation}

        where :math:`c` is the speed of light.

        Parameters:
            voltage (float): The voltage of the electron beam in kV.

        Returns:
            float: The wavelength of the electron beam in Angstroms.
        """
        wavelength = np.sqrt(
            1000.0 * voltage + 0.97845e-6 * np.power(1000.0 * voltage, 2)
        )

        return 12.2639 / wavelength

    def _compute_envelope_function(self, freq_mag: float) -> float:
        """Computes the envelope function for the given absolute frequency
        using the held CTF parameters.

        Parameters:
            freq_mag (float): The magnitude of the spatial frequency component.

        Returns:
            float: The value of the envelope function.
        """
        return np.exp(-0.25 * self.B_factor * freq_mag * freq_mag)

    def compute_ctf_1D(self, spatial_frequencies: np.ndarray) -> np.ndarray:
        """"""
        freq_mag = spatial_frequencies
        freq_mag2 = freq_mag * freq_mag

        delta_Z = 0.5 * (self.defocus_1 + self.defocus_2)

        chi = (
            delta_Z
            - 0.5 * self.wavelength**2 * self.spherical_aberration * freq_mag2
        )  # noqa: E501
        chi *= np.pi * self.wavelength * freq_mag2
        chi += np.arctan(
            self.amplitude_contrast_ratio
            / np.sqrt(1 - self.amplitude_contrast_ratio**2)
        )

        ctf_arr = -np.sin(chi)

        return ctf_arr

    def compute_ctf_2D(self, shape: tuple[int, int]) -> np.ndarray:
        """Shape is assumed to be the shape of an image in Fourier space whose
        pixel spacing values are the same as the held pixel size.
        """
        freq_mag = _calculate_pixel_spatial_frequency(shape, self.pixel_size)
        freq_arg = _calculate_pixel_frequency_angle(shape, self.pixel_size)

        freq_mag2 = freq_mag * freq_mag

        delta_Z = self.defocus_1 + self.defocus_2
        delta_Z += (self.defocus_1 - self.defocus_2) * np.cos(
            2 * (freq_arg - self.astigmatism_azimuth)
        )
        delta_Z *= 0.5

        chi = (
            delta_Z
            - 0.5 * self.wavelength**2 * self.spherical_aberration * freq_mag2
        )  # noqa: E501
        chi *= np.pi * self.wavelength * freq_mag2
        chi += np.arctan(
            self.amplitude_contrast_ratio
            / np.sqrt(1 - self.amplitude_contrast_ratio**2)
        )

        ctf_arr = -np.sin(chi)

        return ctf_arr

    def apply_ctf(self, image: np.ndarray) -> np.ndarray:
        """Apply the CTF to an image. Assumes the pixel size of the image is
        the same as the held pixel size.

        Args:
            image (np.ndarray): The image to apply the CTF to.

        Returns:
            np.ndarray: The image with the CTF applied, in real space.
        """
        ctf_2D = self.compute_ctf_2D(image.shape)

        image_tmp = np.fft.fft2(image)
        image_tmp = np.fft.fftshift(image_tmp)
        image_tmp *= ctf_2D
        image_tmp = np.fft.ifftshift(image_tmp)
        image_tmp = np.fft.ifft2(image_tmp)

        return image_tmp.real

    def to_json(self) -> dict:
        """Convert the ContrastTransferFunction object to a JSON-serializable
        dictionary.
        """
        return {
            "voltage": self.voltage,
            "spherical_aberration": self.spherical_aberration
            * self.pixel_size
            / 1e7,  # convert back to mm
            "defocus_1": self.defocus_1,
            "defocus_2": self.defocus_2,
            "astigmatism_azimuth": self.astigmatism_azimuth,
            "amplitude_contrast_ratio": self.amplitude_contrast_ratio,
            "B_factor": self.B_factor,
            "pixel_size": self.pixel_size,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "ContrastTransferFunction":
        """Create a ContrastTransferFunction object from a JSON dictionary."""
        return cls(
            voltage=json_dict["voltage"],
            spherical_aberration=json_dict["spherical_aberration"],
            defocus_1=json_dict["defocus_1"],
            defocus_2=json_dict["defocus_2"],
            astigmatism_azimuth=json_dict["astigmatism_azimuth"],
            amplitude_contrast_ratio=json_dict["amplitude_contrast_ratio"],
            B_factor=json_dict["B_factor"],
            pixel_size=json_dict["pixel_size"],
        )
