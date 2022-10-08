# 정재현

## clustering

### K-means Clustering

우리는 cluster가 3개임을 알고있다. (픽토그램, 일러스트레이션, 스케치) 그리고 여기에 실제 사진, 또는 다름 사진이 noise로 들어있다. 

### Agglomerative clustering

### Hierarchical Clustering and Dendrograms

### DBSCAN

## evaluate clustering

DBSCAN은 cluster에 포함되지 않는 데이터를 얻을 수 있다.

### ARI

나쁘다고 판단. 실제로 필토그램, 일러스트레이션, 스케치에 대한 타켓 정보가 없으므로 실제 사진을 추출하는데 사용할 수 없다.

### NMI

ARI와 같은 이유로 나쁘다고 판단.

### robustness-based

### silhouette coefficient

## f1-score

우선 f1-score를 이해하기 위해서는 TP, TN, FP, FN의 개념을 이해할 필요가 있다.<br/><br/>

**TP (True Positive) : Positive로 예측한 데이터 중에 실제로 Positive인 수**<br/>
**TN (True Negative) : Negative로 예측한 데이터 중에 실제로 Negative인 수**<br/>
**FP (False Positive) : Positive로 예측한 데이터 중에 실제로 Negative인 수**<br/>
**FN (False Negative) : Negitive로 예측한 데이터 중에 실제로 Positive인 수**<br/>

**즉, FP, FN은 잘못 예측한 경우이댜.**

> 여기서 Positive,와 Negative로 설명했다. 하지만 개를 Positive, 고양이를 Negative로 결정하는 방식으로 사용이 가능하다.
> 더 나아가, 0과 9의 숫자 이미지를 예측한다 생각해보자. 그리고 0을 Positive로 결정하면, 나머지 숫자는 Negative로 결정된다. 그리고 1을 Positive로 결정할 때도 동일하다.

### Precision

**Precision의 정의는 다음과 같다.**<br/>
$Precision = \dfrac{TP}{TP + FP}$<br/>
**Precision은 Positive로 예측한 데이터 중에, 실제로 Positive인 데이터이다.**

### Recall

**Recall의 정의는 다음과 같다.**<br/>
$Recall = \dfrac{TP}{TP + FN}$<br/>
**Recall은 실제 Positive 중에서, Positive로 예측한 데이터이다.**

### Precision과 Recall의 의미

그렇다면 Precision과 Recall은 왜 중요한가.

### f1-score

$F1 = \dfrac{2 \times Precision \times Recall }{Precision + Recall}$<br/><br/>
