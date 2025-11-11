curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
. $HOME/.cargo/env

curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env

mkdir $HOME/train
cd $HOME/train

uv venv dtrain --python 3.12
. $HOME/train/dtrain/bin/activate 

git clone https://github.com/PanocularAI/torchtitan.git
git clone https://github.com/PanocularAI/torchft.git

uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
cd $HOME/train/torchtitan
uv pip install -r requirements.txt

cd $HOME/train/torchft
uv pip install .

. $HOME/train/dtrain/bin/activate

cd $HOME/train/torchtitan
rm -rf /path/to/outputs
python launch.py --module torchtitan.train --job.config_file /path/to/model.toml
