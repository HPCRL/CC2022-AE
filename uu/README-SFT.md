Prerequisite: gcc >=8.5.0; cuda_11.3.0; cudnn v8.2.0;


1) Get source code
git clone --recursive … cc-ae.
cd cc-ae
git submodule sync
git submodule update --init --recursive

2) Create a conda virtual environment and dependencies
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install psutil

3) Build Pytorch from source code:
python setup.py develop

4) Set environment to use uu package:
export UU=$your folder path..$/cc-ae   
export PYTHONPATH=$UU

You have to build the code with cuda and cudnn enabled. We currently do not support cpu or no-cudnn operators.

5) Here is the code structure:
uu/benchmarking := It contains all benckmarks python files and bash script to generate data for plotting fig 8 in paper;It has 2 CNN networks and comparison codes(default pytorch and pytorch-checkpoint); It also has 2 correctness check python files to ensure that our work(SFT) produces exact same output tensor after forward pass and gradient tensors after backpropogation for each conv2d layers.
uu/layers := Our customized operators.
uu/utils := Some dependencies file of SFT.

6) How to extend SFT to other models:
I use "vgg-16-our.py" as an exmaple template to explain how we implemnt SFT to a CNN network.
a) cnn network define:
Line 15-129: Define a CNN network by using our customized operators(e.g. conv2d.TiledConv2d, maxpool2d.cMaxPool2d). ALl argument of the customized operators are closed to default nn operators. Our work relies on recomputaion to save GPU device memory. The recomputation segments are defined in a "sequential.mSequential" structure in Line 124.

b) define network forward computation:
In our work, we split a huge input image into multiple some pieces and execute smaller input one by one. The general structure is Line 140-160. Since we need to gather output after loop structure, we difine an empty output tensor before loop structure in Line 138 and the shape is infered by Line 134 based on the segment structure defined in a).
You can connect any other default torch.nn oeprator after our network like in Line 161-164.

c) The general forward and backward process is same as defualt pytorch code. It is defined in main() function in Line 168-206.





