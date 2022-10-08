# TrajGen: Generating Realistic and Diverse Trajectories With Reactive and Feasible Agent Behaviors for Autonomous Driving
Implementation of the [TrajGen][website_arxiv] and I-SIM simulator based on [Interaction Dataset][website_INTER]. TrajGen is a two-stage trajectory generation framework, which can capture realistic and diverse behaviors directly from human demonstration.

[website_arxiv]: https://sites.google.com/view/trajgen/
[website_INTER]: http://www.interaction-dataset.com/

<img width="48%" src="https://github.com/gaoyinfeng/TrajGen/blob/main/gif/trajgen_1.gif"> <img width="48%" src="https://github.com/gaoyinfeng/TrajGen/blob/main/gif/trajgen_2.gif">

- [Project website](https://sites.google.com/view/trajgen/)
- [Research paper](https://arxiv.org/pdf/2203.16792.pdf)

If you find this code useful, please reference in your paper:

```
@article{zhang2022trajgen,
  title={TrajGen: Generating Realistic and Diverse Trajectories With Reactive and Feasible Agent Behaviors for Autonomous Driving},
  author={Zhang, Qichao and Gao, Yinfeng and Zhang, Yikang and Guo, Youtian and Ding, Dawei and Wang, Yunpeng and Sun, Peng and Zhao, Dongbin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  doi={10.1109/TITS.2022.3202185}
}
```

## Manual Instructions
To properly run TrajGen on your system, you should clone this repository, and follow the instruction below to install the dependencies for both I-SIM and TrajGen.
### 1. Dependencies for I-SIM simulator
Since the HD maps of Interaction Dataset uses the format of Lanelet2, you needs to build Lanelet2 Docker first:
```sh
git clone https://github.com/fzi-forschungszentrum-informatik/Lanelet2
cd Lanelet2
docker build -t isim .
```
After build, run docker container and do port mapping, we use port 5557-5561 as example:
```sh
docker run -it -e DISPLAY -p 5557-5561:5557-5561 -v $path for TrajGen$:/home/developer/workspace/interaction-dataset-master -v /tmp/.X11-unix:/tmp/.X11-unix --user="$(id --user):$(id --group)" --name isim57 isim:latest bash
```


### 2. Dependencies for TrajGen

## Usage
This repo has contained ..., if you want to ... by yourself, check ...
the workspace of I-SIM docker container
