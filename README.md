# TPGSR
Text Prior Guided Scene Text Image Super-Resolution

![alt text](https://github.com/mjq11302010044/TPGSR/visualization/TextSupReso-vis_sr_v2.png)

1. Environment:
```
GPU: GTX1080TI or GTX2080TI
Python >= 3.6
Pytorch >= 1.2
Numpy and other possible python packages
```

2. Download the pretrained recognizer from: 

	Aster: https://github.com/ayumiymk/aster.pytorch
	
	MORAN: https://github.com/meijieru/crnn.pytorch
	
	CRNN: https://github.com/Canjie-Luo/MORAN_v2

3. Unzip the codes and walk into the '$TPGSR_ROOT$/', place the pretrained weights from recognizer in '$TPGSR_ROOT$/src'.

4. Run the train-prefixed shell to train the corresponding model (e.g. TPGSR-TSRN):
```
chmod a+x train_TPGSR-TSRN.sh
./train_TPGSR-TSRN.sh
```
5. Run the test-prefixed shell to test the corresponding model.
```
Adding '--go_test' in the shell file
```
