# Programs List

本文档简要介绍了MPicker中包含的所有脚本。关于每个脚本的具体用法和可用参数，请通过运行 `Mpicker_xxx.py --help` 查看内置帮助信息。

## Core functions

* **Mpicker_gui.py**：启动MPicker图形用户界面（GUI）的脚本。

* **Mpicker_core_gui.py**：由MPicker GUI在提取膜表面和进行膜展平时调用的脚本，也可单独使用。

* **Mpicker_memseg.py**：由MPicker GUI在进行膜分割时调用的脚本，也可单独使用。

* **Mpicker_epicker.py**：由MPicker GUI在使用EPicker进行训练和颗粒挑选时调用的脚本，也可单独使用。

* **Mpicker_epicker_batch.py**：对多套展平断层图数据使用EPicker进行颗粒挑选，可避免重复启动EPicker，从而提高效率。

* **Mpicker_add_coord.py**：以坐标文件的形式向展平断层图中导入颗粒坐标信息，功能与 GUI中的 “Load” 类似。

* **Mpicker_convert_coord.py**：在展平断层图中的颗粒坐标与原始断层图中的颗粒坐标之间进行转换。

## Class2D and STA

* **Mpicker_particles.py**：导出MPicker GUI中所有展平断层图的颗粒信息到一个文件。可用于后续二维分类的计算。

* **Mpicker_2dprojection.py**：根据颗粒的坐标和方向从断层图中提取二维颗粒投影图像及对应的2dCTF图像，用于后续二维分类计算。此外，还可生成平均三维密度图（不考虑CTF）。

* **Mpicker_2dprojection_torch.py**：与`Mpicker_2dprojection.py`类似，但使用GPU（依赖 PyTorch）进行加速，计算速度更快，尤其对于大颗粒。

* **Mpicker_3dctf.py**：根据倾角范围、离焦值等信息生成简单的3dCTF文件，用于根据中心截面定理获取2dCTF。

* **Mpicker_class2d.py**：一个用于二维分类的独立程序，采用与Relion相似的EM算法，支持自定义2dCTF，并依赖PyTorch实现高效计算。

* **Mpicker_align_class2d.py**：对二维分类得到的二维平均结果进行对齐。如果同一种颗粒被分成多个类别，该脚本可使这些类别具有一致的位置和取向。

* **Mpicker_convert_2dto3d.py**：将二维分类结果转换为三维颗粒坐标和欧拉角，用于后续子断层平均计算。建议直接使用`Mpicker_convert_class2d.py`。

* **Mpicker_convert_class2d.py**：功能与`Mpicker_convert_2dto3d.py`类似，但使用更加方便，可同时处理多个类别结果以及多套断层图数据。

* **Mpicker_prepare_rln2.py**：将`Mpicker_convert_class2d.py`的输出结果转换为Relion2或3.0可直接使用的`particles.star`文件，也可用于准备其它子断层平均所需的输入文件。

* **Mpicker_prepare_rln4.py**：将`Mpicker_convert_class2d.py`的输出结果转换为 Relion4或5可直接使用的star文件。

* **Mpicker_convert_tilt90.py**：根据颗粒的坐标和取向生成star文件，并将膜的法向量旋转至X轴方向而非Z轴（与Relion5一致）。建议直接使用`Mpicker_prepare_rln4.py`。

## Flattening based on mesh

* **Mpicker_generatemesh.py**：将提取的膜表面（点云数据，mrc文件或MPicker生成的npz文件）转换为三角网格obj文件。

* **Mpicker_meshparam.py**：对三角网格进行初始的曲面参数化，作为运行`Mpicker_optcuts.py`前的预处理步骤。

* **Mpicker_optcuts.py**：调用OptCuts对三角网格进行曲面参数化，自动对曲面进行切割，以保证展平后的整体畸变足够小。

* **Mpicker_flattenmesh.py**：基于三角网格（obj文件）进行膜展平，并将展平断层图导入 MPicker GUI。所使用的三角网格需已完成曲面参数化（包含纹理坐标）。

* **Mpicker_generatemesh_closed.py**：根据膜分割结果生成封闭的膜表面（类似Chimera中的等值面），并保存为obj文件。

## Other functions

* **Mpicker_autoextract.py**：从膜分割结果中自动提取膜表面，并导入（或直接创建）MPicker GUI。可选择在提取后自动执行膜展平。

* **Mpicker_check.py**：一个用于查看mrc文件的独立程序，类似于IMOD的XYZ mode，但支持多层截面的叠加显示。

* **Mpicker_merge.py**：将断层图中的任意两个XY截面以不同颜色通道进行叠加显示。

* **Mpicker_convert_mrc.py**：将npz格式的被提取的膜表面文件转化为mrc文件，也可将mrc文件转化为npz文件。

* **Mpicker_mrcnpy2mrc.py**：根据原始断层图和展平断层图的几何信息（npy文件）重新生成展平断层图，可用于在不同原始断层图中严格展平相同区域。

* **Mpicker_npy2area.py**：根据展平断层图的几何信息（npy文件）计算每个像素对应的实际物理面积。在无畸变的理想情况下，所有像素的结果应为1。

* **Mpicker_putparticles.py**：在给定所有颗粒的三维坐标和欧拉角以及颗粒密度图的情况下，将颗粒摆回三维空间，生成一个合并后的大mrc文件。
