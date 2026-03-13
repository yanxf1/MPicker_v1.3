# Release Note

## MPicker version 1.3.0, 2025-12-18
- Added `Mpicker_merge.py` to merge two slices in a flattened tomogram with different color channel.
- Added `Mpicker_autoextract.py` to automatically extract and flatten surfaces from membrane segmentation.
- Added a function in GUI to flatten all selected surfaces in batch.
- Added `Mpicker_class2d.py` as a standalone program to do 2D classification (requires PyTorch). The algorithm is similar to Relion, but it supports 2dCTF and is faster. A tutorial of Class2D was also provided.
- Updated the environment, added the library `cython` in `env.yml` and `env_simple.yml`, see **Installation** for details.
- Added the **Programs List** in our website to briefly introduce all `Mpicker_xxx.py` scripts.

## MPicker version 1.2.0, 2025-2-13
- Optimized the function for projection on elliptical cylinder. Set `Project Order` to `-2` to force the use of circular cylinder (elliptical cylinder remains `-1`).
- Surface ID can be shown when right-clicking on the "purple curves".
- Added `Mpicker_2dprojection_torch.py`, the GPU version of `Mpicker_2dprojection.py`, which accelerates 2D particle extraction, especially for large sizes. It can also efficiently sum the sub-volume by setting `--tomoout`.
- Added `Mpicker_align_class2d.py` to automatically align the result of Class2D.
- Added `Mpicker_prepare_rln2.py` and `Mpicker_prepare_rln4.py` to further simplify the preparation of STA files (run after `Mpicker_convert_class2d.py`).
- Fixed a bug where Euler angle conversion may be incorrect in some special cases.
- Updated the **Installation** (for OptCuts) and **Advanced Tutorial**.
- Updated the environment, added libraries `igl` and `opt_einsum` in `env.yml` and `env_simple.yml`, see **Installation** for details.
- Fixed compatibility issues with higher versions of PyTorch (tested with PyTorch 1.13.1).
- Other minor changes.
- Fixed some bugs.
- Packages for installation
    - [MPicker_code_v1.2.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_v1.2.0.tar.gz)
    - [MPicker_code_noseg_v1.2.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_noseg_v1.2.0.tar.gz)

## MPicker version 1.1.0, 2024-8-21
- Added more functions in the GUI (such as Eraser, FlipZ, Screenshot...).
- Sped up 3D display by using texture.
- The surface in 3D mode can be exported as obj file (with texture).
- Optimized the usage of EPicker in GUI.
- Optimized the usage of `Mpicker_epicker_batch.py`.
- Added `Mpicker_convert_class2d.py` to convert the result from Class2D (more powerful than `Mpicker_convert_2dto3d.py`).
- Added more options in `Mpicker_particles.py` (such as output in star file format).
- Added `Mpicker_check.py` as a standalone tool to open tomogram (same as the xyz mode in GUI).
- Other minor changes.
- Fixed some bugs.
- Packages for installation
    - [MPicker_code_v1.1.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_v1.1.0.tar.gz)
    - [MPicker_code_noseg_v1.1.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_noseg_v1.1.0.tar.gz)

## MPicker version 1.0.0, 2023-12-25
- The initial version of MPicker.
- Packages for installation
    - [MPicker_code_v1.0.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_v1.0.0.tar.gz)
    - [MPicker_code_noseg_v1.0.0.tar.gz](http://thuem.net/website/MPicker/Packages/MPicker_code_noseg_v1.0.0.tar.gz)
