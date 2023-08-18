# Learning from Untrimmed Videos: Self-Supervised Video Representation Learning with Hierarchical Consistency
[Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN&authuser=1), [Shiwei Zhang](https://www.researchgate.net/profile/Shiwei-Zhang-14), [Ziyuan Huang](https://huang-ziyuan.github.io/), [Yi Xu](https://scholar.google.com/citations?user=D4jEMqEAAAAJ&hl=en), [Xiang Wang](https://scholar.google.com/citations?user=cQbXvkcAAAAJ&hl=zh-CN&oi=sra), Mingqian Tang,
[Rong Jin](https://www.cse.msu.edu/~rongjin/), [Changxin Gao](https://scholar.google.com/citations?user=4tku-lwAAAAJ&hl=zh-CN), [Nong Sang](https://scholar.google.com/citations?user=ky_ZowEAAAAJ&hl=zh-CN) <br/>
In CVPR, 2022. [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/).

# Running instructions
To train the model with HiCo, set the `_BASE_RUN` to point to `configs/pool/run/training/simclr.yaml`. See `configs/projects/hico/simclr_*_s3dg.yaml` for more details. Alternatively, you can also find some pre-trained model in the `MODEL_ZOO.md`.

For detailed explanations on the approach itself, please refer to the [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/).

For an example run, set the `DATA_ROOT_DIR` and `ANNO_DIR` in `configs/projects/hico/simclr_hacs_s3dg.yaml`, and `OUTPUT_DIR` in `configs/projects/hico/pt-hacs/s3dg-hico-s.yaml`, and run the command for the short pre-training:
```
python runs/run.py --cfg configs/projects/hico/pt-hacs/s3dg-hico-s.yaml
```
Run this command for the long pre-training:
```
python runs/run.py --cfg configs/projects/hico/pt-hacs/s3dg-hico-l.yaml
```

<br/>
<div align="center">
    <img src="HiCo.png" width="350px" />
</div>
<br/>

# Citing HiCo
If you find HiCo useful for your research, please consider citing the paper as follows:
```BibTeX
@inproceedings{qing2022hico,
  title={Learning from Untrimmed Videos: Self-Supervised Video Representation Learning with Hierarchical Consistency},
  author={Qing, Zhiwu and Zhang, Shiwei and Huang, Ziyuan and Xu, Yi and Wang, Xiang and Tang, Mingqian and Gao, Changxin and Jin, Rong and Sang, Nong},
  booktitle={{CVPR}},
  year={2022}
}
```