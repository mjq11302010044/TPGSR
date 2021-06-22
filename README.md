# TPGSR
Text Prior Guided Scene Text Image Super-Resolution


1. Download the pretrained recognizer from: 
	Aster: https://github.com/ayumiymk/aster.pytorch
	MORAN: https://github.com/meijieru/crnn.pytorch
	CRNN: https://github.com/Canjie-Luo/MORAN_v2

2. Unzip the codes and walk into the '$TPGSR_ROOT$/src', place the pretrained weights from recognizer in '$TPGSR_ROOT$/src'.

3. Run the train-prefixed shell to train the corresponding model.
4. Run the test-prefixed shell to test the corresponding model.