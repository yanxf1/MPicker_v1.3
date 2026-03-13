# 2D Classification

这个教程会介绍如何使用`MPicker_class2d.py`进行二维分类（后续简称为Class2D）的计算。它使用了和Relion类似的EM算法，但是支持自定义的2dCTF而且速度更快。它依赖PyTorch库，并且可以使用GPU进行加速。`MPicker_class2d.py`的输出格式与Relion类似，所以这里选择使用Relion的程序`relion_display`来可视化二维分类的结果。

如果希望尝试完整的流程，包括从tomogram中抠取颗粒投影并获得2dCTF，计算二维分类，以及将二维分类的结果转换为后续子断层平均可用的文件，请在[Download](https://thuem.net/software/mpicker/download.html)中下载完整的教程文件`MPicker_tutorial_class2d_v1.3.0.tar.bz2` (3G)。如果仅尝试二维分类本身，可以下载更小的教程文件`MPicker_tutorial_class2d_part_v1.3.0.tar.bz2` （300M），它不包含文件夹`class2d_full_workflow`。

首先在一个空文件夹中解压教程文件：
```bash
tar -jxvf MPicker_tutorial_class2d_v1.3.0.tar.bz2

ls # should contain 3 folders: class2d_full_workflow, class2d_SPA, class2d_tomo
```

## Class2D of SPA Data

首先从常规的单颗粒分析的数据开始处理。这里使用了Relion3教程（ftp://ftp.mrc-lmb.cam.ac.uk/pub/scheres/relion30_tutorial.pdf）中的beta-galactosidase数据进行测试。一共约有7000个颗粒，每个颗粒的尺寸为64\*64 pixels。

首先进入文件夹`class2d_SPA`。其中`particles.star`是记录了颗粒信息的star文件（与Relion的格式相同），`Movies`是存放颗粒文件的文件夹，`cmd.txt`记录了将要使用的命令。

`particles.star`中最重要的一列是`rlnImageName`（红色），它记录了颗粒文件（mrcs）的位置，必须提供。

`rlnMicrographName`在这里被用于颗粒的分组。事实上，`MPicker_class2d.py`会按照`rlnOpticsGroup`、`rlnGroupName`、`rlnMicrographName`的优先级进行分组。如果都不提供则认为所有颗粒属于一个group。

`rlnMagnification`和`rlnDetectorPixelSize`决定了颗粒的pixel size，所有颗粒需要保持一致。如果不提供则需要在后续执行`MPicker_class2d.py`时显示指定pixel size。 

`rlnVoltage`等信息（黄色）会被用于计算每个颗粒的CTF，如果不提供则在二维分类时不考虑CTF。此外，也可以使用`rlnCtfImage`来指定每个颗粒的2dCTF文件（mrcs）的位置，并直接使用2dCTF作为每个颗粒的CTF，后续处理tomogram的投影颗粒时会用到。

其它star文件中的信息不会被使用。

为了使用`MPicker_class2d.py`计算二维分类，请执行如下命令。我们使用了一块2080 Ti GPU，计算时间约为**3min**。
```bash
Mpicker_class2d.py -i particles.star -o m220k50/run -d 220 -k 50 -g 0 # -p 3.54 -n 25 --normalize

relion_display --gui # choose m220k50/run_it025_model.star
```
其中`-i`指定了输入的star文件，`-o`指定了输出文件的前缀，`-d`指定了mask的直径，`-k`指定了分类的类数。

`-g`用于指定使用的GPU的id（不指定则默认使用CPU），如果需要使用多个GPU可以设定如`-g 0,1,2`。但由于这里颗粒尺寸较小，颗粒数量较少，且采样间隔较大，使用多个GPU的提升并不显著。 

`-p`用于指定pixel size，但由于`particles.star`中已经包含了此信息所以可以跳过。`-n`用于设定迭代轮数，默认值为25轮。`--normalize`指定程序根据颗粒的mean和std进行归一化预处理，但这里使用的颗粒在抠取时已经做过归一化处理，所以可以跳过。

为了查看结果，可以运行`relion_display --gui`，然后选择文件`m220k50/run_it025_model.star`，并且勾选`Sort images on rlnClassDistribution`和`Reverse sort`，就可以得到类似下图的结果。或者也可以通过`relion_display --i m220k50/run_it025_classes.mrcs`的方式直接查看分类结果。

![class2d_spa](images/class2d_spa.jpg)

此外需要指出，`_data.star`文件中记录的`rlnNrOfSignificantSamples`为根据颗粒概率分布（包括类别、平移、旋转）计算出的信息熵的指数，即`int(2^sum(-P*logP))`，与Relion中的定义并不相同。

## Class2D of Tomogram Projections

这里首先会对衣藻tomogram中的PSII沿着膜法向量进行投影，然后通过二维分类获得颗粒取向，最后将二维分类结果转换为可直接用于Relion中子断层平均的文件。

教程使用的数据来自[EMPIAR-12469](https://www.ebi.ac.uk/empiar/EMPIAR-12469/)。为了减少文件大小，文件夹`class2d_full_workflow`中只包含部分tomogram文件用于展示流程（共包含约1600个颗粒）。文件夹`class2d_tomo`中包含了完整的颗粒文件（约5300个），但不包含tomogram，仅用于测试二维分类部分的效果。二维分类之外的部分，在[Advanced Tutorial](https://thuem.net/software/mpicker/tutorial_advance.html#get-input-files-for-class2d)中的**Get Input Files for Class2D**和**Process the Result from Class2D**也有进行讲解。

- ### About Tutorial Files

首先进入文件夹`class2d_full_workflow`。其包含的内容如下。
* 文件夹`subtomo`包含了三个小的tomogram，并按照类似于Relion 3.0计算子断层平均的格式进行组织。
* 文件`tomo_part*_data.txt`记录了每个tomogram的颗粒坐标与角度，可通过`Mpicker_particles.py`对MPicker GUI的挑选结果进行导出而得到。它包含6列数字，分别是每个颗粒的x、y、z、rot、tilt、psi，由于膜的法向量只能决定tilt和psi，所以这里rot都为0。
* 文件`cmd.txt`记录了将要使用的命令。
* 文件`mrcs2data.txt`和`mrcs2tomo.txt`是在进行数据转换时需要用到的文件。用于建立颗粒投影文件（mrcs）、颗粒取向文件（txt）和tomogram名字之间的关联。
* 文件`prepare_rln2.txt`可帮助准备Relion 2或Relion 3.0计算子断层平均所需的文件，而不必使用官方的基于`relion_prepare_subtomograms.py`的流程。它包含6行数字：Voltage(in KV), Cs(in mm), AmpContrast, PixelSize(in A), UseOnlyLowerTiltDefociLimit(in degree), Bfactor。

![class2d_files](images/class2d_files.jpg)

- ### Prepare Files for Class2D

与[Advanced Tutorial](https://thuem.net/software/mpicker/tutorial_advance.html#get-input-files-for-class2d)中类似，首先使用`Mpicker_3dctf.py`生成每个tomogram的3dCTF文件。这里的三个tomogram实际来自于同一个大tomogram，所以可以共用一个`3dctf.mrc`文件。这里`--cos`代表会根据倾角的cos值降低高角度区域的强度，与Relion类似。`--box 70`代表3dCTF的尺寸为70\*70\*70像素。

然后使用`Mpicker_2dprojection_torch.py`，根据颗粒的坐标与取向（`--data`），从tomogram文件（`--map`）和3dCTF文件（`--ctf`）中提取出颗粒投影文件（`--output`）和2dCTF文件（`--ctfout`）。提取结果会被记录在star文件（`--star`）中用于后续的二维分类。`--dxy`代表颗粒投影的尺寸，需要与3dCTF的尺寸保持一致。`--dz`代表投影的深度，约为PSII膜外区的厚度。 `--invert`代表需要对输出的颗粒进行反衬度操作，保证颗粒为白色。`--tomoout`是可选的，可输出一个平均的三维密度图，帮助判断颗粒中心以及膜的位置。具体命令如下。

```bash
Mpicker_3dctf.py --df 58381 --pix 7.26 --box 70 --t1 -58.36 --t2 55.07 --cos --out 3dctf.mrc

Mpicker_2dprojection_torch.py --map subtomo/Tomograms/tomo_part1/tomo_part1.mrc --ctf 3dctf.mrc \
--data tomo_part1_data.txt --dxy 70 --dz 7 --invert --gpuid 0 \
--output 2dproj_tomo_part1.mrcs --ctfout 2dctf_tomo_part1.mrcs --star class2d_merge.star \
--tomoout sum_part1.mrc
```

随后，继续处理剩下的两个tomogram，`tomo_part2.mrc`和`tomo_part3.mrc`。由于这里需要把结果追加在star文件的尾部，所以增加了参数`--conti`。具体命令如下。

```bash
Mpicker_2dprojection_torch.py --map subtomo/Tomograms/tomo_part2/tomo_part2.mrc --ctf 3dctf.mrc \
--data tomo_part2_data.txt --dxy 70 --dz 7 --invert --gpuid 0 \
--output 2dproj_tomo_part2.mrcs --ctfout 2dctf_tomo_part2.mrcs --star class2d_merge.star \
--tomoout sum_part2.mrc --conti

Mpicker_2dprojection_torch.py --map subtomo/Tomograms/tomo_part3/tomo_part3.mrc --ctf 3dctf.mrc \
--data tomo_part3_data.txt --dxy 70 --dz 7 --invert --gpuid 0 \
--output 2dproj_tomo_part3.mrcs --ctfout 2dctf_tomo_part3.mrcs --star class2d_merge.star \
--tomoout sum_part3.mrc --conti
```
![class2d_prepare](images/class2d_prepare.jpg)

- ### Class2D

随后使用汇总了所有颗粒信息的`class2d_merge.star`计算二维分类。这里分10类，迭代20轮，pixel size 为7.26 A，mask直径为260 A。我们使用了一块2080 Ti GPU，计算时间不到半分钟。

相比于SPA数据，对tomogram投影进行二维分类更加困难，且容易过拟合。所以这里使用了`--mask_noise`参数，以使用相位随机化噪声填充颗粒mask之外的区域（默认使用0填充）。此外`--T`降低为1（默认为2），以使结果更加平滑。此外，也可以尝试通过设置`--skip_mask_ref`参数，来跳过对reference施加mask的步骤，有时可以获得更好的结果。

由于获得颗粒投影时没有进行归一化，所以这里使用了`--normalize`参数根据颗粒的mean和std进行归一化。使用`--load_in_memory`可以在最开始将所有颗粒文件读入内存，节省文件IO的时间。`--seed`可以指定随机数种子使结果可重复。

具体命令如下。

```bash
Mpicker_class2d.py -i class2d_merge.star -o m260k10/run -k 10 -n 20 -g 0 -p 7.26 -d 260 \
--mask_noise --T 1 --normalize --load_in_memory --seed 42

# Mpicker_class2d.py -i class2d_merge.star -o m260k10_skiprefmask/run -k 10 -n 20 -g 0 -p 7.26 \
# -d 260 --mask_noise --T 1 --normalize --load_in_memory --skip_mask_ref --seed 42

relion_display --i m260k10/run_it020_classes.mrcs

# relion_display --gui # m260k10/run_it020_model.star
```

二维分类后，可以使用Relion的`relion_display`查看二维平均的结果`_classes.mrcs`。如果希望按照颗粒占比进行排序，可以通过`relion_display --gui`打开GUI，并选择文件`_model.star`。结果如下。

![class2d_result](images/class2d_result.jpg)

- ### from Class2D to STA

由于只使用了很少的颗粒，这里分类结果并不十分理想。为了方便演示，这里假设红框标记的3个类（也就是第4、第9、第10类）都是目标颗粒。具体流程与[Advanced Tutorial](https://thuem.net/software/mpicker/tutorial_advance.html#process-the-result-from-class2d)中类似。注意实际计算结果可能与图中不完全相同，所以请根据实际情况判断好颗粒对应的类别。

由于3个好颗粒的二维平均之间并不能互相对齐，首先需要通过`Mpicker_align_class2d.py`对二维平均进行对齐，并生成`fmove.txt`文件用于描述每个类的平移与旋转修正。也可以手动对齐并设置每个类别的平移与旋转修正，`fmove.txt`仅包含4列数据：类别编号（rlnClassNumber）、x位移（in pixel）、y位移（in pixel）、旋转角度（in degree）。

具体命令如下。`--mo`是可选项，用于输出对齐后的二维平均结果。`--one`代表类别编号是从1开始，而不是0（如THUNDER）。`--ids`代表需要保留的好类的编号，`--ref`代表用作reference的类的编号，这里使用了最清晰的一类。`--refxy`代表reference的中心坐标，in pixel，以IMOD坐标系为准，但从0开始。

```bash
Mpicker_align_class2d.py --i m260k10/run_it020_classes.mrcs --o fmove.txt --mo aligned.mrcs \
--one --ids 4,9,10 --ref 10 --refxy 36,34

relion_display --i aligned.mrcs
```

然后就可以使用`Mpicker_convert_class2d.py`将二维分类的结果`_data.star`转化为颗粒的三维坐标和欧拉角了。具体命令如下。其中`--data`包含每个tomogram在抠取颗粒投影时使用的坐标与角度信息。`--star`是二维分类的结果。`--out2`是转换后的三维坐标与欧拉角。`--out`用于汇总每个颗粒的二维分类结果与抠取颗粒投影时所用的参数，不进行换算，一般不需要输出。

```bash
Mpicker_convert_class2d.py --data mrcs2data.txt --star m260k10/run_it020_data.star \
--fmove fmove.txt --out2 good_particles.txt
```

![class2d_align](images/class2d_convert.jpg)

由于二维分类会修正颗粒在膜表面上的位移，所以`good_particles.txt`还包含`rlnOrigin[XYZ]`的信息。与Relion相同，`rlnCoordinateX - rlnOriginX`为修正后的X坐标。

- ### Prepare STA Files for Relion

为了进一步简化准备STA文件的流程，MPicker还提供了准备Relion所需文件的脚本。`Mpicker_prepare_rln2.py`可以获得Relion2（或3.0）所需的文件。`Mpicker_prepare_rln4.py`可以获得Relion4（或5）所需的文件。根据我们的经验，对较为困难的蛋白，先使用Relion2（或3.0）计算，再使用Relion4进一步refine可以得到更好的结果。尽管Relion2使用起来不如Relion4方便。

为了准备**Relion2**所需的文件，可以使用如下命令。其中`--i`是刚才得到的，记录颗粒坐标与欧拉角的文件。`--l`包含每个tomogram的名字（也就是之后`Tomograms`文件夹中每个tomogram对应的文件夹名）。`--p`为pixel size，可以写入最终的`particles.star`。`--o`代表用于存放输出文件（particles.star, tomo_part1.coords, tomo_part2.coords, tomo_part3.coords）的文件夹。

如果提供了`--prepare_all`文件，则可以跳过官方流程中`relion_prepare_subtomograms.py`的部分。此时`--o`文件夹中需要包含`Tomograms`文件夹，其中每个tomogram对应一个文件夹，每个文件夹中需要包含`.defocus`、`.mrc`、`.order`三个文件。最后`Mpicker_prepare_rln2.py`会把每个tomogram的坐标存成`.coords`文件。

```bash
Mpicker_prepare_rln2.py --i good_particles.txt --l mrcs2tomo.txt \
--o subtomo --p 7.26 --prepare_all prepare_rln2.txt
```

![class2d_rln2](images/class2d_rln2.jpg)

然后就可以按照Relion2的官方流程，计算每个颗粒的3dCTF，抠取颗粒（job alias需设置为"extract_tomo"），并使用MPicker生成的`particles.star`作为输入计算Refine3D或Class3D。Relion2的官方使用流程可以参考"https://doi.org/10.1038/nprot.2016.124"或者"https://www3.mrc-lmb.cam.ac.uk/relion/index.php?title=Sub-tomogram_averaging"。

![class2d_rln2_recon](images/class2d_rln2_recon.jpg)

直接使用**Relion4**进行计算更加方便，因为其支持在颗粒文件中直接指定每个颗粒的坐标和取向。具体命令如下。`--i`和`--l`的含义与之前相同。`--o`为包含所有颗粒信息的star文件。`--s`代表坐标的缩放倍数，因为Relion4通常使用Bin1的数据作为输入，而这里使用的tomogram实际为Bin2的。

```bash
Mpicker_prepare_rln4.py --i good_particles.txt --l mrcs2tomo.txt \
--o subtomo/particles_rln4.star --s 2

# Mpicker_prepare_rln4.py --i xxx_data.star --o particles_rln4.star --s 2 --from_rln2
```

![class2d_rln4](images/class2d_rln4.jpg)

此外，如果开启`--from_rln2`，`Mpicker_prepare_rln4.py`也可用于将Relion2中的计算结果转化为Relion4支持的颗粒文件。

## Class2D of All Projections

这里使用全部的PSII投影颗粒（约5300个）计算二维分类，可以得到比之前更好的结果。

进入文件夹`class2d_tomo`，并执行如下命令。`--ctf_thres`用于判断哪些区域属于missing wedge，其默认值为0.001。由于这里的2dCTF来自于Relion生成的3dCTF，在missing wedge区域有较大的波动，所以这里使用了更大的阈值。设置`--ctf_thres -1`可以关闭这个功能，它对结果的影响一般不是非常显著。

```bash
Mpicker_class2d.py -i class2d_full.star -o m260k10_skiprefmask/run -k 10 -n 20 -g 0 \
-p 7.26 -d 260 --mask_noise --T 1 --normalize --load_in_memory \
--skip_mask_ref --seed 42 --ctf_thres 0.05

relion_display --gui &
```

我们使用了一块2080 Ti GPU，计算时间约为**1min**。

![class2d_tomo](images/class2d_tomo.jpg)
