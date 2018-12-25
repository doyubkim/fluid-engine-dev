set -e

sudo apt-get -yq update
sudo apt-get -yq install build-essential cmake libtbb-dev ffmpeg
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
. ~/.bashrc
conda install -y matplotlib
echo "backend: agg" > `python -c "import matplotlib; print(matplotlib.matplotlib_fname())"`
git clone https://github.com/doyubkim/fluid-engine-dev.git --recursive
cd fluid-engine-dev
mkdir build
cd build
cmake ..
make -j `nproc` manual_tests
./bin/manual_tests
python ../scripts/render_manual_tests_output.py
