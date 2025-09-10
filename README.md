# RAIDENでの実行メモ

ssh utsu@raiden.riken.jp

qrsh -jc gpu-container_g1_dev -ac d=nvcr-pytorch-2503

export http_proxy=http://10.1.10.1:8080
export https_proxy=http://10.1.10.1:8080
export ftp_proxy=http://10.1.10.1:8080

cd ~/sp
uv venv
source .venv/bin/activate

uv pip install torch torchvision numpy pandas matplotlib scikit-learn pyyaml wilds pot

python main.py




scp -r "/mnt/c/Users/utsunomiya/OneDrive - 筑波大学/@k_pro/sp/"* utsu@raiden.riken.jp:~/sp/

scp -r utsu@raiden.riken.jp:~/sp/results "/mnt/c/Users/utsunomiya/OneDrive - 筑波大学/@k_pro/sp/"
