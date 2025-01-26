# 케창딥> Chpter 02 _ 신경망의 수학적 구성 요소

### 이 장에서 다를 핵심 내용]

- 신경망 예제 만들기
- 텐서와 텐서 연산의 개념
- 역전파와 경사 하강법을 사용하여 신경망이 학습되는 방법

## 2-1. 신경망과의 첫 만남

```python
전체코드 미리보기.

from tensorflow.keras.datasets import minst
 (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
from tensorflow import keras
from kensorflow.keras import layers

model = keras.Sequential([
		layers.Dense(512, activation = 'relu'),
		layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'rmsprop',
							loss = 'sparse_categorical_crossentropy',
							metrics = ['accuracy'])
							
train_images = train_images.reshape((6000, 28 * 28))
train_images = train_images.astype('float32') / 255,

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

model.fit(train_images, train_labels, epoch = 5, batch_size = 128)

test_digitals = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]
```

```python
코드 2-1 케라스에서 MNIST 데이터 셋 적제하기

 from tensorflow.keras.datasets import minst
 (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

- train_images, train_labels가 모델이 학습해야할 훈련 세트(train set)
- test_images, test_labels가 테스트 세트(test set).

```python
코드 2-2 신경망 구조

from tensorflow import keras
from kensorflow.keras import layers

model = keras.Sequential([
		layers.Dense(512, activation = 'relu'),
		layers.Dense(10, activation = 'softmax')
])
```

- 신경망의 핵심 구성 요소는 층.
- 대부분의 딥러닝은 간단한 층을 연결한여 구성.
- 점진적으로 데이터를 정제하는 형태.
- 완전 연결(Fully Connected)된 신경망 층인 Dense층 2개가 연속되어 있음.
- 마지막층은 10개의 확률 점수가 들어있는  배열을 반환, Softmax분류 층임.
- 숫자가 10개의 이미지 클래스 중 하나에 속할 확율을 출력.
- 신경망의 훈련 준비를 마치기 위해 컴파일 단계에 포함되어야 할 3가지.
    - [**옵티마이저(Optimizer)](https://chatgpt.com/share/67963a6b-7874-8007-9f06-daf7937ad0c9)** : 성능을 향상시키기 위해 입력 된 데이터를 기반으로 모델을 업데이 
                                           트 하는 메커니즘
    - [손실 함수(Loss Function)](https://chatgpt.com/share/67963dc1-20ec-8007-9513-e420cfb94d1b) : 훈련 데이터에서 모델의 성능을 측정하는 방법으로 모델이 옳은 
                                              방향으로 학습될 수 있도록 도와줌.
    - 훈련과 테스트 과정을 모니터링할 지표 : 여기에서는 정확도(분류된 이미지의 비율)만 
                                                                     고려함.
    

```python
코드 2-3 컴파일 단계

model.compile(optimizer = 'rmsprop',
							loss = 'sparse_categorical_crossentropy',
							metrics = ['accuracy'])
```

- 훈련 전 데이터를 모델에 맞는 크리고 변경, 모든 값을 0과 1사이로 스케일 조정.
- 훈련 이미지는 [0, 255] 사이의 값인 uint8타입 (60000, 28, 28) 크기를 가진 배열로 저장되어 있음.
- 이를 0과 1 사이의 값을 가지는 float32 타입의 (6000, 28 * 28) 크기인 배열로 변경.

```python
코드 2-4 이미지 데이터 준비하기.

train_images = train_images.reshape((6000, 28 * 28))
train_images = train_images.astype('float32') / 255,

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```

- 여기까지 모델을 훈련시킬 준비가 끝.
- model.fit() 메서드를 호출하여 훈련 모델에 모델을 학습 진행

```python
코드 2-5 모델 훈련하기

model.fit(train_images, train_labels, epoch = 5, batch_size = 128)
```

- loss(손실)과 accuracy(정확도) 2개의 정보가 출력.
- 훈런 데이터에 대해 0.9 이상의 정확도를 보임.
- 이렇게 훈련된 모델을 사용해 새로운 숫자 이미지에 대한 클레스 확률을 예측해 보자.

```python
코드 2-5 모델을 사용하여 예측 만들기

test_digitals = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]
```

- 출력된 배열 인덱서  i에 있는 숫자는 숫자 이미지 test_digits[0]이 클래스 i에 속할 확률에 해당
- 인덱서 7에 대해 가장 높은 확률 값을 얻음. ( 9.999)
- 예측 결과는 ‘ 7 ‘
- 이전에 본 적 없는 숫자애 대한 분류 시도.
- 평균적인 정확도 계산 시도

```python
코드 2-7 새로운 데이터에서 모델 평가하기.

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'테스트 정확도 : {test_acc}')
```

- 정확도 약 0.979.
- 훈련 테스트 정확도 0.989보다 아주 약간 낮음.
- 훈련 정확도와 테스트 정확도 사이의 차이는 [과대적합(Overfitting)](https://chatgpt.com/share/67967412-47cc-8007-a561-48822812f950) 때문.
