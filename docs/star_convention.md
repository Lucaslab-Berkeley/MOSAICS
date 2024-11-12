# MOSAICS `.star` file conventions

The STAR ((cite)) file format is ubiquitous across cryo-EM for its ability to define and share tabular metadata about images and processing pipelines.
MOSAICS adopts the STAR format for loading and serializing metadata associated with experiments with conventions roughly following that of RELION ((reference relion documentation)).
This custom STAR format attempts to define all necessary metadata and file references for setting up a MOSAICS experiment rather than spreading information across multiple decoupled files.
Label names for each data block, short definitions, and their associated labels in the RELION format are described below.

## Optics group
The optics group defines parameters of the microscope, the micrograph & image stack shapes & pixel sizes, and fit parameters for the contrast transfer function.
Fields roughly follow RELION, but additional labels have been added to account for additional necessary information.
The modifications are as below:
```
rlnOpticsGroupName             --> mosaicsOpticsGroupName
rlnOpticsGroup                 --> mosaicsOpticsGroupNumber
rlnMicrographOriginalPixelSize --> mosaicsMicrographOriginalPixelSize
                         (new) --> mosaicsMicrographOriginalShapeX
                         (new) --> mosaicsMicrographOriginalShapeY
                         (new) --> mosaicsMicrographPSDReference
rlnVoltage                     --> mosaicsVoltage
rlnSphericalAberration         --> mosaicsSphericalAberration
rlnAmplitudeContrast           --> mosaicsAmplitudeContrast
                         (new) --> mosaicsAdditionalPhaseShift
                         (new) --> mosaicsDefocus1
                         (new) --> mosaicsDefocus2
                         (new) --> mosaicsAstigmatismAzimuth
rlnImagePixelSize              --> mosaicsImagePixelSize
rlnImageSize                   --> (removed)
                         (new) --> mosaicsImageShapeX
                         (new) --> mosaicsImageShapeY
                         (new) --> mosaicsImageShapeZ
rlnImageDimensionality         --> mosaicsImageDimensionality
rlnCtfDataAreCtfPremultiplied  --> (removed)
```

Fields, associated units & descriptions as well as default values for the optics data block are defined in the table below

| MOSAICS STAR label                   | type    | units    | required/default    | description |
|--------------------------------------|---------|----------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mosaicsOpticsGroupName`             | string  | none     | micrograph basename | String defining the optics group name, default is the basename of the micrograph the optics group references.                                                            |
| `mosaicsOpticsGroupNumber`           | integer | none     | `0`                 | Unique integer defining the index of the optics group.                                                                                                                   |
| `mosaicsMicrographOriginalPixelSize` | float   | Angstrom | required            | Pixel size of the original micrograph, in units of Angstroms.                                                                                                            |
| `mosaicsMicrographOriginalShapeX`    | integer | pixels   | automatic           | Shape of the reference micrograph along the X axis (slow axis). Automatically inferred from the micrograph array during instantiation.                                   |
| `mosaicsMicrographOriginalShapeY`    | integer | pixels   | automatic           | Shape of the reference micrograph along the Y axis (fast axis). Automatically inferred from the micrograph array during instantiation.                                   |
| `mosaicsMicrographPSDReference`      | string  | none     | `None`              | Reference path (absolute) to the pre-calculated and saved numpy .txt file containing the power spectral density values and associated spatial frequencies. Not required. |
| `mosaicsVoltage`                     | float   | kV       | `300.0`             | Microscope voltage.                                                                                                                                                      |
| `mosaicsSphericalAberration`         | float   | mm       | `2.7`               | Microscope spherical aberration.                                                                                                                                         |
| `mosaicsAmplitudeContrast`           | float   | unitless | `0.07`              | Microscope amplitude contrast ratio.                                                                                                                                     |
| `mosaicsAdditionalPhaseShift`        | float   | radians  | `0.0`               | Additional phase shift applied globally to contrast transfer function.                                                                                                   |
| `mosaicsDefocus1`                    | float   | Angstrom | required            | Major defocus value of the CTF for the micrograph, in Angstroms.                                                                                                         |
| `mosaicsDefocus2`                    | float   | Angstrom | required            | Minor defocus value of the CTF for the micrograph, in Angstroms.                                                                                                         |
| `mosaicsAstigmatismAzimuth`          | float   | degrees  | required            | Astigmatism azimuth of the CTF, in units of degrees, relative to the x-axis.                                                                                             |
| `mosaicsImagePixelSize`              | float   | Angstrom | automatic           | Pixel size of the particle image stack, in units of Angstroms *Note: currently inferred as the same as the original micrograph.*                                         |
| `mosaicsImageShapeX`                 | integer | pixels   | automatic           |                                                                                                                                                                          |
| `mosaicsImageShapeY`                 | integer | pixels   | automatic           |                                                                                                                                                                          |
| `mosaicsImageShapeZ`                 | integer | count    | automatic           |                                                                                                                                                                          |
| `mosaicsImageDimensionality`         | integer | none     | 3                   |                                                                                                                                                                          |

The optics group data block is closely related to the `OpticsGroups` class ((TODO, add API reference)) which can parse/export the information in this data block while exposing it as an indexable Python object.

## Particles group
Each particle has associated orientation, location, and micrograph membership information which is important for defining the MOSAICS workflow.
Again, labels roughly follow the RELION format with additional fields for pre-processed information important for high-resolution template matching.

```
rlnCoordinateX    --> mosaicsCoordinateXPixel
rlnCoordinateY    --> mosaicsCoordinateYPixel
            (new) --> mosaicsCoordinateXAngstrom
            (new) --> mosaicsCoordinateYAngstrom
            (new) --> mosaicsParticleDefocusAngstrom
rlnClassNumber    --> mosaicsParticleClassNumber
            (new) --> mosaicsParticleClassName
rlnRot            --> mosaicsOrientationPhi
rlnTilt           --> mosaicsOrientationTheta
rlnPsi            --> mosaicsOrientationPsi
            (new) --> mosaicsIndexInImageStack
rlnImageName      --> mosaicsImagePath
rlnMicrographName --> mosaicsMicrographPath
rlnOpticsGroup    --> mosaicsOpticsGroupName
rlnGroupNumber    --> mosaicsOpticsGroupNumber
rlnOriginXAngst   --> (removed)
rlnOriginYAngst   --> (removed)
```

Fields, associated units & descriptions as well as default values for the particles data block are defined in the table below

| MOSAICS STAR label               | type    | units    | required/default    | description |
|----------------------------------|---------|----------|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| `mosaicsCoordinateXPixel`        | integer | pixels   | required            | The x pixel coordinate of the particle in the original micrograph. Position reference is the center of the particle box.  |
| `mosaicsCoordinateYPixel`        | integer | pixels   | required            | The y pixel coordinate of the particle in the original micrograph. Position reference is the center of the particle box.  |
| `mosaicsCoordinateXAngstrom`     | float   | Angstrom | required            | The y coordinate of the particle in physical units of Angstroms. Position reference is the center of the particle box.    |
| `mosaicsCoordinateYAngstrom`     | float   | Angstrom | required            | The x coordinate of the particle in physical units of Angstroms. Position reference is the center of the particle box.    |
| `mosaicsParticleDefocusAngstrom` | float   | Angstrom | required            | The defocus value of the particle relative to the global CTF of the micrograph defined in the optics group.               |
| `mosaicsParticleClassNumber`     | integer |          | `0`                 | Optional integer defining class membership of the particle. Default of `0` implies all particles are of the same class.   |
| `mosaicsParticleClassName`       | string  |          | `None`              | Optional string descriptor for particle class. Default is `None` for each particle implying no data on class description. |
| `mosaicsOrientationPhi`          | float   | degrees  | required            | Orientation of the particle in units of degrees. First rotation along the Z axis in ZYZ convention.                       |
| `mosaicsOrientationTheta`        | float   | degrees  | required            | Orientation of the particle in units of degrees. Second rotation along the Y' axis in ZYZ convention.                     |
| `mosaicsOrientationPsi`          | float   | degrees  | required            | Orientation of the particle in units of degrees. Third rotation along the Z'' axis in ZYZ convention.                     |
| `mosaicsIndexInImageStack`       | integer |          | required            | Index of the particle in the image stack `.mrcs` file.                                                                    |
| `mosaicsImagePath`               | string  |          | optional (see note) | Path to the stack of cropped particle images `.mrcs` file.                                                                |
| `mosaicsMicrographPath`          | string  |          | optional (see note) | Path to the original micrograph.                                                                                          |
| `mosaicsOpticsGroupName`         | string  |          |                     | Optics group name.                                                                                                        |
| `mosaicsOpticsGroupNumber`       | integer |          |                     | Optics group number.                                                                                                      |

*Note: Only one of `mosaicsImagePath` and `mosaicsMicrographPath` are optional. The particle image stack is loaded from `mosaicsImagePath`, if provided. But if `mosaicsMicrographPath` is provided and `mosaicsImagePath` is not, then the particle images will automatically be extracted from the micrograph upon instantiation.*

MOSAICS defines particle orientations using the Euler angles $(\phi, \theta, \psi)$ in the ZYZ convention.
The origin of the micrograph is assumed to be the top left-hand corner of the micrograph with particle coordinates referencing the center of particle.

The particles data block is closely related to the `ParticleStack` class ((TODO, add API reference)) which parses and exports this tabular data.
