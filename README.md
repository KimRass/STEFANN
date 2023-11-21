# 'STEFANN' (Roy et al., 2020) implementation from scratch in PyTorch
## Average SSIM
## Implementation Details
<!-- ### Dataset Split -->
<!-- - 기존에 'fannet/valid' 디렉토리에 있던 300개의 폰트 중 20%를 test set으로, 나머지는 validation set으로 분리했습니다. ('dataset/fannet').
    ```bash
    # e.g.,
    python3 dataset/split_dataset.py\
        --src_fannet_dir="/Users/jongbeomkim/Documents/datasets/stefann/fannet/fannet"
    ```
- 이로써 train set, validation set, test set은 각각 1015, 240, 60개의 폰트를 갖습니다. -->
### FANnet
- 공식 저장소에서는 TensorFlow를 사용했지만 저는 PyTorch를 사용해 구현했습니다.
- 데이터셋에서는 숫자와 알파벳 대소문자가 있지만 논문과 공식 저장소에서는 26개의 알파벳 대문자만을 사용해 학습시켰습니다. 저는 62개의 모든 문자를 사용해 학습시켰습니다.
- 논문과 공식 저장소에서는 one-hot encoded label에 fully-connected layer를 사용했지만 저는 embedding layer를 사용했습니다. 이 레이어 다음에는 ReLU activation function을 사용하지 않는 편이 모델의 성능이 더 좋았습니다.
    ```python
    self.label_embed = nn.Embedding(N_CLASSES, dim)
    ```
- 논문과 공식 저장소에서는 마지막 레이어에서 ReLU activation function을 사용했지만 저는 input tensor를 $[-1, 1]$로 normalize했기 때문에 hyperbolic tangent를 사용했습니다.
- 논문과 공식 저장소에서는 L1 loss를 사용했지만 ("The network minimizes the mean absolute error (MAE).") 저는 L2 loss를 사용했습니다. L2 loss가 L1 loss보다 수렴 속도가 더 빨랐습니다.
- Instance normalization을 사용해 봤지만 오히려 학습이 잘 이루어지지 않았습니다.
<!-- - 공식 저장소에서는 fully-connected layer 다음에도 ReLU activation function을 사용했지만 저는 convolutional layer 다음에만 사용했습니다. -->
## Theoretical Background
### SSIM (Structural SIMilarity)
$$\text{SSIM}(x, y) = \frac{(2\mu_{x}\mu{y} + c_{1})(2\sigma + c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2} + \sigma_{y}^{2} + c_{2})}$$
## `libraqm` Installation
- For the error saying `KeyError: 'setting text direction, language or font features is not supported without libraqm'`, run `pip install --upgrade Pillow  --global-option="build_ext" --global-option="--enable-raqm"`
## References
- https://github.com/prasunroy/stefann
