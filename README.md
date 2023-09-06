# Disentangling Spatial and Temporal Learning for Efficient Image-to-Video Transfer Learning
[Zhiwu Qing](https://scholar.google.com/citations?user=q9refl4AAAAJ&hl=zh-CN&authuser=1), [Shiwei Zhang](https://www.researchgate.net/profile/Shiwei-Zhang-14), [Ziyuan Huang](https://huang-ziyuan.github.io/), [Yingya Zhang], [Changxin Gao](https://scholar.google.com/citations?user=4tku-lwAAAAJ&hl=zh-CN),
[Deli Zhao],  [Nong Sang](https://scholar.google.com/citations?user=ky_ZowEAAAAJ&hl=zh-CN) <br/>
In ICCV, 2023. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/coming_soon.pdf).

<br/>
<div align="center">
    <img src="framework.jpg" />
</div>
<br/>

# Latest

[2023-09] Codes are available!

This repo is a modification on the [TAdaConv](https://github.com/alibaba-mmai-research/TAdaConv) repo.
## Installation

Requirements:
- Python>=3.6
- torch>=1.5
- torchvision (version corresponding with torch)
- simplejson==3.11.1
- decord>=0.6.0
- pyyaml
- einops
- oss2
- psutil
- tqdm
- pandas

optional requirements
- fvcore (for flops calculation)

# Model Zoo

We include our pre-trained models in the [MODEL_ZOO.md](MODEL_ZOO.md).


# Running instructions
You can find some pre-trained model in the `MODEL_ZOO.md`.

For detailed explanations on the approach itself, please refer to the [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/).

For an example run, set the `DATA_ROOT_DIR` and `ANNO_DIR` in `configs/projects/dist/vit_base_16_ssv2.yaml`, and `OUTPUT_DIR` in `configs/projects/dist/ssv2-cn/vit-b16-8+16f_e001.yaml`, and run the command for fine-tuning:
```
python runs/run.py --cfg configs/projects/dist/ssv2-cn/vit-b16-8+16f_e001.yaml
```



# Citing HiCo
If you find DiST useful for your research, please consider citing the paper as follows:
```BibTeX
@inproceedings{qing2023dist,
  title={Disentangling Spatial and Temporal Learning for Efficient Image-to-Video Transfer Learning},
  author={Qing, Zhiwu and Zhang, Shiwei and Huang, Ziyuan and Yingya Zhang and Gao, Changxin and Deli Zhao and Sang, Nong},
  booktitle={ICCV},
  year={2023}
}
```
