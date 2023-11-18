# STEFANN (Roy et al., 2020) implementation from scratch in PyTorch
## Dataset Split
```bash
# e.g.,
ython3 dataset/split_dataset.py\
    --src_fannet_dir="/Users/jongbeomkim/Documents/datasets/stefann/fannet/fannet"
```
## Implementation Details
### FANnet
- 공식 저장소에서는 TensorFlow를 사용했지만 저는 PyTorch를 사용해 구현했습니다.
- 데이터셋에서는 숫자와 알파벳 대소문자가 있지만 논문과 공식 저장소에서는 26개의 알파벳 대문자만을 사용해 학습시켰습니다. 저는 62개의 모든 문자를 사용해 학습시켰습니다.
## References
- https://github.com/prasunroy/stefann
