# MPicker (v1.3)
Welcome to use MPicker. It's a tool for membrane flatten and visualization in the field of cryoET.

See our website `http://thuem.net` for details: [Installation](https://thuem.net/software/mpicker/installation.html), [Tutorial](https://thuem.net/software/mpicker/tutorial.html), [Download](https://thuem.net/software/mpicker/download.html).

For installation, try `requirements.txt` if the `env.yml` is too old for your machine.

## Only interested in 2D classification?

Just copy the `Mpicker_class2d.py` and `mpicker_class2d_utils.py` in `mpicker_gui`. The required libraries are: PyTorch, NumPy, mrcfile, opt_einsum, and tqdm. 

See the tutorial [2D Classification](https://thuem.net/software/mpicker/tutorial_class2d.html) for details.

## Programs List

This is a brief overview of all scripts included in MPicker. For detailed usage and available options, please refer to the built-in help by running `Mpicker_xxx.py --help`.

### Core functions

* **Mpicker_gui.py**: Launches the MPicker graphical user interface (GUI).

* **Mpicker_core_gui.py**: Called by the MPicker GUI for membrane surface extraction and membrane flattening. It can also be used as a standalone script.

* **Mpicker_memseg.py**: Called by the MPicker GUI for membrane segmentation. It can also be used as a standalone script.

* **Mpicker_epicker.py**: Called by the MPicker GUI to perform training and particle picking using EPicker. It can also be used as a standalone script.

* **Mpicker_epicker_batch.py**: Performs particle picking on multiple flattened tomograms, can increase efficiency by avoiding repeated EPicker startup.

* **Mpicker_add_coord.py**: Loads particle coordinates file into a flattened tomogram, similar to the "Load" function in the GUI.

* **Mpicker_convert_coord.py**: Converts particle coordinates between flattened tomograms and the raw tomograms.

### Class2D and STA

* **Mpicker_particles.py**: Exports all particle information of flattened tomograms into a single file, which can be used for subsequent 2D classification.

* **Mpicker_2dprojection.py**: Extracts 2D particle projection images and corresponding 2dCTF images from tomograms based on particle coordinates and orientations, for subsequent 2D classification. It can also generate averaged 3D density map (CTF not considered).

* **Mpicker_2dprojection_torch.py**: Similar to `Mpicker_2dprojection.py`, but accelerated by GPU (requires PyTorch), providing improved performance especially for big particles.

* **Mpicker_3dctf.py**: Generates a simple 3dCTF file based on tilt angle range, defocus values, and other related parameters. It is used to extract 2dCTFs according to the central slice theorem.

* **Mpicker_class2d.py**: A standalone program for 2D classification. It uses an EM algorithm similar to Relion, but supports custom 2dCTFs, and achieves high computational efficiency with PyTorch acceleration.

* **Mpicker_align_class2d.py**: Aligns the 2D averaged image of 2D classification. If the same type of particle is split into multiple classes, this script enforces consistent positions and orientations across classes.

* **Mpicker_convert_2dto3d.py**: Converts 2D classification results into 3D particle coordinates and Euler angles for subsequent subtomogram averaging.

* **Mpicker_convert_class2d.py**: Similar to `Mpicker_convert_2dto3d.py`, but more convenient to use. It can process multiple classes and multiple tomogram datasets at once.

* **Mpicker_prepare_rln2.py**: Converts the output of `Mpicker_convert_class2d.py` into `particles.star` files directly usable by Relion 2 or Relion 3.0. It can also prepare input files for subsequent subtomogram averaging.

* **Mpicker_prepare_rln4.py**: Converts the output of `Mpicker_convert_class2d.py` into star files directly usable by Relion 4 or 5.

* **Mpicker_convert_tilt90.py**: Generates star files from particle coordinates and orientations while rotating the membrane normal to point along the X axis, instead of the Z axis, following the convention used in Relion 5. The same effect can be achieved using `Mpicker_prepare_rln4.py`.

### Flattening based on mesh

* **Mpicker_generatemesh.py**: Converts extracted membrane surfaces (point cloud data, in npz or mrc format) into triangular meshes (obj file).

* **Mpicker_meshparam.py**: Performs initial surface parameterization of triangular meshes as a preprocessing step before running `Mpicker_optcuts.py`.

* **Mpicker_optcuts.py**: Calls OptCuts to parameterize triangular meshes. It automatically cuts the mesh to ensure low average distortion after flattening.

* **Mpicker_flattenmesh.py**: Performs membrane flattening based on a triangular mesh (obj file) and imports the resulting flattened tomogram into the MPicker GUI. The mesh must have been parameterized in advance (the obj file should include texture coordinates).

* **Mpicker_generatemesh_closed.py**: Generates a closed boundary from a membrane segmentation, similar to the isosurface in Chimera, and saves it as a triangular mesh in obj format.

### Other functions

* **Mpicker_autoextract.py**: Automatically extracts membrane surfaces from a membrane segmentation, and imports them into (or creates) the MPicker GUI. It can also automatically perform membrane flattening after the extraction. The default parameters were set based on our thylakoid data with pixel size about 10 A.

* **Mpicker_check.py**: A standalone GUI for viewing mrc files, similar to the XYZ mode in IMOD, but supporting overlay display of multiple slices (projection).

* **Mpicker_merge.py**: A standalone GUI to overlay any two XY slices from a tomogram using different color channels.

* **Mpicker_convert_mrc.py**: Converts extracted surface files between npz and mrc formats.

* **Mpicker_mrcnpy2mrc.py**: Regenerates a flattened tomogram using the geometry information (npy file) and a raw tomogram. This can be used to strictly flatten the same region across different raw tomograms.

* **Mpicker_npy2area.py**: Computes the actual physical area of each pixel in XY slices, based on the geometry information (npy file) of a flattened tomogram. In the ideal case without any area distortion, the value should be 1 for all pixels.

* **Mpicker_putparticles.py**: Given the 3D coordinates and Euler angles of all particles, and a particle density map, places all particles back into 3D space to generate a single combined mrc file.
