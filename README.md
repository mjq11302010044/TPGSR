# Text Prior Guided Scene Text Image Super-Resolution
https://arxiv.org/abs/2106.15368

_Jianqi Ma, Shi Guo, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)_  
_Department of Computing, [The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

#### Recovering TextZoom samples
![TPGSR visualization](./visualization/TextSupReso-vis_sr_v2.png)

## Environment:


![python](https://img.shields.io/badge/python-v3.7-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.2-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v9.1-green.svg?style=plastic)
![numpy](https://img.shields.io/badge/numpy-1.18-green.svg?style=plastic)

```
Other possible python packages like pyyaml, cv2, Pillow and imgaug
```

## Main idea
### Single stage with loss
<img src="./visualization/TextSupReso-TPGSR_v3.png" width="960px"/> 

### Multi-stage version
<img src="./visualization/mt.jpg" width="480px"> 

## Configure your training
### Download the pretrained recognizer from: 

	Aster: https://github.com/ayumiymk/aster.pytorch  
	MORAN:  https://github.com/Canjie-Luo/MORAN_v2  
	CRNN: https://github.com/meijieru/crnn.pytorch

Unzip the codes and walk into the '$TPGSR_ROOT$/', place the pretrained weights from recognizer in '$TPGSR_ROOT$/'.

### Download the TextZoom dataset:

	https://github.com/JasonBoy1/TextZoom

### Train the corresponding model (e.g. TPGSR-TSRN):
```
chmod a+x train_TPGSR-TSRN.sh
./train_TPGSR-TSRN.sh
```

### Run the test-prefixed shell to test the corresponding model.
```
Adding '--go_test' in the shell file
```
## Cite this paper:

	@article{ma2021text,
  	title={Text Prior Guided Scene Text Image Super-resolution},
  	author={Ma, Jianqi and Guo, Shi and Zhang, Lei},
  	journal={arXiv preprint arXiv:2106.15368},
  	year={2021}
	}

