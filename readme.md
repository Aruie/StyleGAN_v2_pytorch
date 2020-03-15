# StyleGAN V2





# 세부 아키텍쳐


- Generator 
  - LeakyRELU 0.2 사용
  - 모든 레이어에 같은 학습률 적용
  - Const Layer 의 경우 하나로 통일
  - 모든가중치는 표준정규분포에서 추출
  - 모든 Bias 와 Noise Scaling Factor 는 0으로 초기화 ( 1로 초기화하는 Y_s 와 관련된 Bias를 제외하고 )
  - 
  - Mapping Network
    - 모든 입력과 출력의 차원은 512
    - 맵핑 네트워크가 깊을때 높은 학습률 적용시 잘학습이안되어 학습율은 1/00 으로감소

