sudo apt-get -yq update
sudo apt-get -yq install build-essential cmake libtbb-dev libglfw3-dev

wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.dev
wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda-repo-ubuntu1604-9-2-148-local-patch-1_1.0-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda



git clone https://github.com/doyubkim/fluid-engine-dev.git --recursive
cd fluid-engine-dev
