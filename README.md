# <img align="right" src="_static/lighton_small.png" width=60/>Model-Free Episodic Control with Optical Random Features

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Twitter](https://img.shields.io/twitter/follow/LightOnIO?style=social)](https://twitter.com/LightOnIO)

This is the code to reproduce the video of the [**Tackling Reinforcement Learning with the Aurora OPU**](https://medium.com/@LightOnIO/88f3ffff137a) blog post on Medium.

## Requirements

Please install required Python packages with `pip install -r requirements.txt`.

## Reproducing our results

To run the script with the Atari 2600 version of the MsPacman game, execute in the shell: 
```sh
python main.py --env="MsPacman-v0"
```
This creates a video file `bestrun.mp4` of the best recorded run.

To see available commands, use `python main.py --help`.

## <img src="_static/lighton_cloud_small.png" width=120/> Access to Optical Processing Units



To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/

For researchers, we also have a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information.

