# ResNet-CIFAR10

### Introduction

 이곳은 Microsoft Research 팀에서 연구, 발표한 [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) 논문을 읽고 구현하여 CIFAR-10 데이터를 이용해 훈련한 모델을 저장한 저장소 입니다. 심층 신경망이 등장한 직후 기존의 규칙 기반 방식이 가지고 있던 많은 한계점을 극복하면서 더 깊은 신경망은 더 높은 성능을 발휘할 것이라는 기대가 만연해 있었습니다. 하지만 실재로 너무 깊은 신경망은 오히려 더 낮은 성능을 보여주었고 이것은 심층 신경망 학습의 딜레마로 여겨졌습니다. 위의 논문에서는 깊은 신경망이 효과적인 학습을 할 수 있도록 해주는 residual leaening을 처음으로 제안하고 있습니다. 또 이 학습 방법을 사용하여 구현된 ResNets은 깊이의 딜레마를 극복하고 효과적인 학습을 보여준 모델입니다.
    
***Residual Learning***   
데이터는 신경망을 지나며 연산되어 점점 원래의 형태를 잃어가게 됩니다. 특히 모델의 최종 결과값이 입력된 데이터로 부터 너무 다른 형태를 가지게 되면(예컨데 분류모델의 경우) 이러한 현상은 더욱 두드러지게 나타납니다. 깊은 신경망의 경우 이전의 층들을 통과하면서 너무 많이 변형되어버린 데이터가 이후의 층들이 학습될때 필요한 만큼의 정보를 지니고 있지 않게 될 수 있습니다. 그리고 이것은 결과적으로 학습해야하는 error를 의도치 않게 증가시키는 요인이 됩니다.  
![image](https://user-images.githubusercontent.com/35767146/126870194-71d9876d-8f7d-459b-88ae-b897eaa67d24.png)   
하지만 residual learning은 idenriry mapping을 위해 설계된 shortcut 을 통해서 특정 레이어의 출력값을 아무런 연산을 거치지 않고(혹은 선형 mapping을 거친 후) 이후의 더 깊은 신경층으로 도달하게 만들 수 있습니다. 이러한 설계로 인하여 모델의 특정한 층들은 이전의 값으로부터 데이터가 얼마나 변형되었는지 학습할 수 있게 됩니다. 즉 깊어짐으로 인해 발생하는 불필요한 error를 모델 스스로가 컨트롤 할 수 있는 형태로 바꾸어 줌으로써 깊은 모델이 얕은 모델에 비해 무조건적으로 더 많은 error를 감당해야하는 부담을 덜어줄 수 있습니다.


### Dependencies
이 저장소에서는 [TensorFlow](https://www.tensorflow.org/) 2.5.0 버전을 기준으로 작성된 script 들을 저장해 두었습니다. 사용된 데이터는 [tensorflow_datasets](https://www.tensorflow.org/datasets)를 이용해서 다운로드받았습니다. tensorflow_datasets에 대한 더 자세한 정보는 [여기](https://github.com/tensorflow/datasets)를 통해 접하실 수 있습니다. pip를 통해서 python의 패키지를 관리하는 경우 아래의 코드를 통해서 다운받을 수 있습니다.
    pip install tensorflo==2.5.0
    pip install tensorflow-datasets


### Data
훈련을 위해 사용된 데이터는 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)입니다. 논문에서는 4.2 에서 설명하고 있으며 저의 코드는 20 층, 0.27 M 개의 파라미터로 이루어진 모델을 기준으로하여 훈련하였습니다.

### Training
![image](https://user-images.githubusercontent.com/35767146/126870958-ec81fa32-7026-4708-bc0c-9576ace3bb61.png)
![image](https://user-images.githubusercontent.com/35767146/126870975-51bbcf9b-f147-4e05-bc7b-4d2ad986e9c4.png)

훈련은 NVIDIA GeForce RTX 3070 으로 진행되었으며 he uniform initialization을 통해 무작위로 발생한 난수로 부터 훈련되었습니다. 훈련을 진행하기전 전체 트레이닝데이터로 부터 channel wise mean을 구해 whitening 작업을 해주었고 test 데이터에도 동일한 factor를 이용해 whitening 작업을 거쳤습니다. 보여지는 그래프의 가로축은 iteration을 나타냅니다.


### Reference
참고한 논문의 BibTeX는 아래와 같습니다.
'''{.no-highlight}
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
'''

### Contact

작성자 : 박상현
E-mail : hyun06000@gmail.com
Blog : https://davi06000.tistory.com/
