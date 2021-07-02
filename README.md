# Text Prior Guided Scene Text Image Super-Resolution
https://arxiv.org/abs/2106.15368

_Jianqi Ma, Shi Guo, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)_  
_Department of Computing, The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China_

#### Recovery Results
![TPGSR visualization](./visualization/TextSupReso-vis_sr_v2.png)

## Environment:

## Usage

![python](https://img.shields.io/badge/python-v3.7-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.2-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v9.1-green.svg?style=plastic)
![numpy](https://img.shields.io/badge/cuda-v9.1-green.svg?style=plastic)

```
Other possible python packages like pyyaml, cv2, Pillow and imgaug
```

2. Download the pretrained recognizer from: 

	Aster: https://github.com/ayumiymk/aster.pytorch
	
	MORAN: https://github.com/meijieru/crnn.pytorch
	
	CRNN: https://github.com/Canjie-Luo/MORAN_v2

3. Unzip the codes and walk into the '$TPGSR_ROOT$/', place the pretrained weights from recognizer in '$TPGSR_ROOT$/'.

4. Run the train-prefixed shell to train the corresponding model (e.g. TPGSR-TSRN):
```
chmod a+x train_TPGSR-TSRN.sh
./train_TPGSR-TSRN.sh
```
5. Run the test-prefixed shell to test the corresponding model.
```
Adding '--go_test' in the shell file
```
6. **More information to be added...**
