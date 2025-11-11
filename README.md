# MotionRefineNet: Fine-Grained Pose Sequence Smoothing and Refinement (ACM MM 2025)

# Abstract
Capturing human motion with existing monocular estimators often results in large errors when dealing with rare poses, occlusions, truncations, and frame blurring, leading to jitter and long-term drift. Although previous methods have introduced post-processing networks for pose refinement, they struggle to balance global smoothing and fine-grained correction. In this work, we propose MotionRefineNet, which leverages the synergy and complementarity between long- and short-term features in the temporal domain and high- and low-frequency features in the frequency domain to address these challenges. The temporal branch is designed as a hierarchical motion structure to learn multi-time scale features, where long-term features learn motion smoothness, and short-term features capture local rapid changes. The frequency branch employs different frequency band learning strategies based on the degrees of freedom (DoF) of body parts. For body parts with low DoF, the focus is on low-frequency features that represent overall motion trends and regular actions. For body parts with high DoF, we design a filter to adaptively extract useful information from all frequency bands, including subtle motion changes in the high-frequency bands. Extensive experiments on multiple datasets and estimators demonstrate that MotionRefineNet outperforms existing methods in refining 2D, 3D, and SMPL poses, achieving superior pose smoothing and deviation correction.


![image](/demo/1.png)

# Quick Start

1. **Download Datasets**:

  Please refer to [Smoothnet: A plug-and-play network for refining human poses in videos](https://github.com/cure-lab/SmoothNet?tab=readme-ov-file).

2. **Training (an example on aist_vibe_3d)**:

```python
python train.py --cfg configs/aist_vibe_3D.yaml --dataset_name aist --estimator vibe --body_representation 3D --slide_window_size 32
```



# Citation
If you find this useful in your research, please consider citing:
```
@inproceedings{li2025motionrefinenet,
  title={MotionRefineNet: Fine-Grained Pose Sequence Smoothing and Refinement},
  author={Li, Haolun and Liu, Weihuang and Liu, Jiateng and Tang, Zhenhua and Pun, Chi-Man and Miao, Qiguang and Xu, Feng and Gao, Hao},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5--14},
  year={2025}
}
```


# License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.



# Acknowledgements

Our code borrows heavily from [Smoothnet](https://github.com/cure-lab/SmoothNet?tab=readme-ov-file). We thank the author for sharing their wonderful code.
