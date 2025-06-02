<p align="center" width="100%">
</p>

<div id="top" align="center">

Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/abs/2505.24378"> 📑 Paper </a> |
<a href="https://github.com/KongYilun/M3DT"> 🐱 Github Repo </a> |
</h4>

<!-- **Authors:** -->

_**Yilun Kong<sup>1</sup>, Guozheng Ma<sup>2</sup>, Qi Zhao<sup>1</sup>, Haoyu Wang<sup>1</sup>, Li Shen<sup>3</sup>, Xueqian Wang<sup>1</sup>, Dacheng Tao<sup>2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> Tsinghua University,
<sup>2</sup> Nanyang Technological University,
<sup>3</sup> Sun Yat-sen University_


</div>


## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [Acknowledgements](#acknowledgments)


## Overview

Despite recent advancements in offline multi-task reinforcement learning (MTRL) have harnessed the powerful capabilities of the Transformer architecture, most approaches focus on a limited number of tasks, with scaling to extremely massive tasks remaining a formidable challenge. 

In this paper, we propose M3DT, a novel mixture-of-experts (MoE) framework that tackles task scalability by further unlocking the model's parameter scalability. Specifically, we enhance both the architecture and the optimization of the agent, where we strengthen the Decision Transformer (DT) backbone with MoE to reduce task load on parameter subsets, and introduce a three-stage training mechanism to facilitate efficient training with optimal performance. Experimental results show that, by increasing the number of experts, M3DT not only consistently enhances its performance as model expansion on the fixed task numbers, but also exhibits remarkable task scalability, successfully extending to 160 tasks with superior performance.



## Quick Start

Download the dataset Massive MT160 via this [Google Drive link](https://drive.google.com/file/d/14Tp3WLCTq6NEaUnuDD1v2qcGou70Xk8I/view?usp=sharing) and change the dataset path in the following scripts.

When your environment is ready, you could run the project by the following steps:

1. Backbone Training
``` Bash
python stage1_backbone_train.py --prefix_name mt160 --embed_dim 256 --seed 0 --data_path ./mt160_used
```

2. Task Grouping
``` Bash
python stage2_task_grouping_gradient.py --prefix_name mt160 --group_num 48 --seed 0
```
or
``` Bash
python stage2_task_grouping_random.py --prefix_name mt160 --group_num 48 --seed 0
```

3. Expert Training
``` Bash
bash stage2_expert_train_total.sh
```

4. Router Traning
``` Bash
python stage3_router_train.py --prefix_name mt160 --embed_dim 256 --expert_num 48 --seed 0
``` 

## Citation
If you find this work is relevant with your research or applications, please feel free to cite our work!
```
@inproceedings{kong2025M3DT,
    title={Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer},
    author={Yilun Kong and Guozheng Ma and Qi Zhao and Haoyu Wang and Li Shen and Xueqian Wang and Dacheng Tao},
    booktitle={International Conference on Machine Learning},
    year={2025},
}
```

## Acknowledgments

This repo benefits from [DT](https://github.com/kzl/decision-transformer), [PromptDT](https://github.com/mxu34/prompt-dt) and [HarmoDT](https://github.com/charleshsc/HarmoDT). Thanks for their wonderful works!
