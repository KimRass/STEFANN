# 'STEFANN' (Roy et al., 2020) implementation from scratch in PyTorch
## Implementation Details
### Dataset Split
- 기존에 'fannet/valid' 디렉토리에 있던 300개의 폰트 중 20%를 test set으로, 나머지는 validation set으로 분리했습니다. ('dataset/fannet').
    ```bash
    # e.g.,
    python3 dataset/split_dataset.py\
        --src_fannet_dir="/Users/jongbeomkim/Documents/datasets/stefann/fannet/fannet"
    ```
- 이로써 train set, validation set, test set은 각각 1015, 240, 60개의 폰트를 갖습니다.
### FANnet
- 공식 저장소에서는 TensorFlow를 사용했지만 저는 PyTorch를 사용해 구현했습니다.
- 데이터셋에서는 숫자와 알파벳 대소문자가 있지만 논문과 공식 저장소에서는 26개의 알파벳 대문자만을 사용해 학습시켰습니다. 저는 62개의 모든 문자를 사용해 학습시켰습니다.
- 논문과 공식 저장소에서는 one-hot encoded label에 fully-connected layer를 사용했지만 저는 embedding layer를 사용했습니다.
    ```python
    self.label_embed = nn.Embedding(N_CLASSES, dim)
    ```
- 논문과 공식 저장소에서는 마지막 레이어에서 ReLU activation function을 사용했지만 저는 이미지 생성 task임을 고려해 hyperbolic tangent를 사용했습니다.
## Theoretical Background
### SSIM (Structural SIMilarity)
$$\text{SSIM}(x, y) = \frac{(2\mu_{x}\mu{y} + c_{1})(2\sigma + c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2} + \sigma_{y}^{2} + c_{2})}$$
## References
- https://github.com/prasunroy/stefann
