# 'STEFANN: Scene Text Editor using Font Adaptive Neural Network'

# 1. Custom Dataset for Training FANnet
- Existing Dataset:
    - Source: https://github.com/prasunroy/stefann
    - 동일한 폰트 사이즈에 대해서 모든 문자 간의 크기와 높이가 동일한 문제 있습니다.
    <table>
        <thead>
            <tr>
                <th colspan=8>'Abel-Regular'</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/48.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/57.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/97.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/98.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/103.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/106.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/65.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abel-regular/81.jpg" width="40"></th>
            </tr>
        </tbody>
        <thead>
            <tr>
                <th colspan=8>'AbhayaLibre-Bold'</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/48.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/57.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/97.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/98.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/103.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/106.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/65.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/resources/as_is_dataset/abhayalibre-bold/81.jpg" width="40"></th>
            </tr>
        </tbody>
    </table>
- Custom Dataset:
    - 동일한 폰트 사이즈와 높이를 가지고 문자를 실제로 렌더링하여 제작했습니다.
    - 문자 간의 상대적인 크기를 반영합니다.
    <table>
        <thead>
            <tr>
                <th colspan=8>'Abel-Regular'</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/48.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/57.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/97.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/98.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/103.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/106.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/65.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/Abel-Regular/81.jpg" width="40"></th>
            </tr>
        </tbody>
        <thead>
            <tr>
                <th colspan=8>'AbhayaLibre-Bold'</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/48.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/57.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/97.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/98.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/103.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/106.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/65.jpg" width="40"></th>
                <th><img src="https://raw.githubusercontent.com/KimRass/STEFANN/refs/heads/main/dataset/fannet_new/train/AbhayaLibre-Bold/81.jpg" width="40"></th>
            </tr>
        </tbody>
    </table>

## 1) Generating Custom Dataset
```bash
python3 dataset/generate_data.py
```

# 2. Samples
## 1) FANnet
- 독특한 폰트에 대해서는 아무리 많이 학습을 시켜도 이미지를 잘 생성하지 못하는 것으로 확인할 수 있었습니다.
    - <img src="https://github.com/KimRass/STEFANN/assets/67457712/80d3aeb1-1b62-4b89-84ba-5a131f0d2cb4" width="400">
    - <img src="https://github.com/KimRass/STEFANN/assets/67457712/92d3a05e-7d8d-42d6-a8e0-b43b106ef3a6" width="400">

# 3. Average SSIM
- "resources/fannet.pth":
    - 0.5215 on validation set

# 4. Implementation Details
<!-- ### Dataset Split -->
<!-- - 기존에 'fannet/valid' 디렉토리에 있던 300개의 폰트 중 20%를 test set으로, 나머지는 validation set으로 분리했습니다. ('dataset/fannet').
    ```bash
    # e.g.,
    python3 dataset/split_dataset.py\
        --src_fannet_dir="/Users/jongbeomkim/Documents/datasets/stefann/fannet/fannet"
    ```
- 이로써 train set, validation set, test set은 각각 1015, 240, 60개의 폰트를 갖습니다. -->

## 1) Main
- M2 MacBook에서 PyQt5가 작동하지 않아 PyQt6로 변경했습니다.
- 원본 코드에서는 텍스트가 배경보다 밝을 경우 제대로 작동하지 않았는데, 바운딩 박스를 생성하기 전에 Tab 키를 통해 이미지를 반전시킨 후 사용하면 잘 작동하도록 코드를 수정했습니다.

## 2) FANnet
- 공식 저장소에서는 TensorFlow를 사용했지만 저는 PyTorch를 사용해 구현했습니다.
- 데이터셋에서는 숫자와 알파벳 대소문자가 있지만 논문과 공식 저장소에서는 26개의 알파벳 대문자만을 사용해 학습시켰습니다. 저는 62개의 모든 문자를 사용해 학습시켰습니다.
- 논문과 공식 저장소에서는 one-hot encoded label에 fully-connected layer를 사용했지만 label encoding과 embedding layer를 사용했습니다. 이 레이어 다음에는 ReLU activation function을 사용하지 않는 편이 모델의 성능이 더 좋았습니다.
    ```python
    self.label_embed = nn.Embedding(N_CLASSES, dim)
    ```
- 논문과 공식 저장소에서는 마지막 레이어에서 ReLU activation function을 사용했지만 저는 input tensor를 $[-1, 1]$로 normalize했기 때문에 hyperbolic tangent를 사용했습니다.
- 논문과 공식 저장소에서는 L1 loss를 사용했지만 ("The network minimizes the mean absolute error (MAE).") 저는 L2 loss를 사용했습니다. L2 loss가 L1 loss보다 수렴 속도가 더 빨랐습니다.
- Instance normalization을 사용해 봤지만 오히려 학습이 잘 이루어지지 않아 제외했습니다.

# 5. Theoretical Background

## 1) SSIM (Structural SIMilarity)
$$\text{SSIM}(x, y) = \frac{(2\mu_{x}\mu{y} + c_{1})(2\sigma + c_{2})}{(\mu_{x}^{2} + \mu_{y}^{2} + c_{1})(\sigma_{x}^{2} + \sigma_{y}^{2} + c_{2})}$$

### (1) `libraqm` Installation
- For the error saying `KeyError: 'setting text direction, language or font features is not supported without libraqm'`, run `pip install --upgrade Pillow  --global-option="build_ext" --global-option="--enable-raqm"`

# 6. References
- [STEFANN Official repostory](https://github.com/prasunroy/stefann)
