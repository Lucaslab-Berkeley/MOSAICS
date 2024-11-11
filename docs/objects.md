# Classes and Serialization in MOSAICS

MOSAICS is an object-oriented Python package with separate objects for dealing with cryo-EM data, microscope parameters, and image processing filters.
Each of these objects is serializable, and we define methods for parsing data into MOSAICS so experiments and workflows can easily be shared and re-run.
Pre-existing data can be loaded into MOSAICS using the following sets of data:
 - Experimental micrograph(s) + reference structure along with *cis*TEM `make_template_result` files
 - Experimental micrograph(s) + reference structure with particle locations and orientations for each micrograph
 - `.star` file for a particle stack and references to original experimental micrograph(s). The conventions for this STAR file are described further down this page.
 - Custom MOSAICS `.star` file defining all the requisite data and parameters. Conventions are again listed below

## Experimental data classes

### The `Micrograph` class

The core object for parsing and working with experimental cryo-EM data is the `Micrograph` class.
A micrograph object can be instantiated from a `.mrc` file or from a numpy array.

```
from mosaics.data_structures import Micrograph

# Micrograph created from a pre-existing mrc file
mrc_path = "/some/path/to/micrograph.mrc"
example_micrograph = Micrograph.from_mrc(mrc_path)

# Micrograph created from a numpy array
arr = np.random.normal(0, 1, (256, 256))
example_micrograph = Micrograph(image_array=arr, pixel_size=1.65)
```

Each instance of a `Micrograph` holds a reference to
 1. An experimental cryo-EM image,
 2. Optional path to the cryo-EM image (`None` if not provided),
 3. The pixel size of the image (in units of Angstroms),
 4. Optional contrast transfer function parameters for the image, and
 5. The optional power spectral density of the image.
All of these parameters are necessary for performing two-dimensional template matching (2DTM) with further information about the contrast transfer function and power spectral density being held under their own classes.

<!-- The parameters for the contrast transfer function can either be added to the micrograph as a pre-existing `ContrastTransferFunction` class, or the parameters can automatically be parsed from a CTFFIND5 (CITE) fit result as shown below

```
ctffind5_result_path = "/another/path/to/some/diagnostic.txt"
example_micrograph.set_ctf_from_diagnostic(ctffind5_result_path)
``` -->

See the API reference (TODO) for further information on the `Micrograph`.

### The `ContrastTransferFunction` class

Defocus parameters for a `Micrograph` are stored in the `ContrastTransferFunction` class.
Currently, the contrast transfer function is implemented as described in CTFFIND5 ((TODO: CITE))

### The `PowerSpectralDensity` class

### The `ParticleStack` class

Cropped particle views are extracted from their original micrograph(s) and placed into a `ParticleStack` object which deals with the images of these particles, their locations, orientations, and cross-correlation scores.
There are multiple ways to create a `ParticleStack` object, the most common being either from a micrograph plus an *cis*TEM `out_coordinates.txt` file or from a `.star` and `.mrcs` file.
Examples of these can be seen in the examples gallery (TODO: link).

Currently, MOSAICS uses a custom `.star` file format to load in / export a particle stack detailed below:

```
data_particle_stack_attributes

_mosaicsPixelSize #1
_mosaicsBoxSize #2
_mosaicsVoltage #3
_mosaicsSphericalAberration #4
_mosaicsAmplitudeContrastRatio #5
_mosaicsBFactor #6
1.06	384	300	2.7	0.07	60.0

data_particle_stack_table

loop_
_mosaicsParticleIndex #1
_mosaicsParticleClass #2
_mosaicsPixelCoordinateX #3
_mosaicsPixelCoordinateY #4
_mosaicsOrientationPhi #5
_mosaicsOrientationTheta #6
_mosaicsOrientationPsi #7
_mosaicsDefocus1 #8
_mosaicsDefocus2 #9
_mosaicsDefocusAngle #10
_mosaicsImageStackPath #11
_mosaicsMicrographPath #12
_mosaicsMicrographPSDPath #13
0	1	30	40	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack0.mrcs	/example/path/micrograph0.mrc	/example/path/micrograph0_psd.txt
1	1	50	60	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack0.mrcs	/example/path/micrograph0.mrc	/example/path/micrograph0_psd.txt
2	1	70	80	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack0.mrcs	/example/path/micrograph0.mrc	/example/path/micrograph0_psd.txt
0	0	90	100	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack1.mrcs	/example/path/micrograph1.mrc	/example/path/micrograph1_psd.txt
1	1	110	120	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack1.mrcs	/example/path/micrograph1.mrc	/example/path/micrograph1_psd.txt
2	1	130	140	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack1.mrcs	/example/path/micrograph1.mrc	/example/path/micrograph1_psd.txt
3	1	150	160	120.0	45.9	32.7	5200.0	4900.0	22.2	/example/path/stack1.mrcs	/example/path/micrograph1.mrc	/example/path/micrograph1_psd.txt
```

*Note: The particle index column refers to the image index in the .mrcs file. This allows multiple stacks to be combined, but there is no checking to ensure particles and files are properly ordered*


## Reference structures

### The `ReferenceStructure` class

## Serialization

Data and general workflows are serialized using the ubiquitous cryo-EM `.star` file format for metadata and `.mrc` files for image type data.
