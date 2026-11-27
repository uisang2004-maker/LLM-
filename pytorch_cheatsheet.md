# PyTorch 정리표: `nn.` 과 `torch.`

『밑바닥부터 만들면서 배우는 LLM』 2~4장과 Do it! 실습(NSMC 감성 분류, KLUE 개체명 인식)에서 쓴 PyTorch 함수를 정리한 표입니다. "어디서" 칸은 처음 만났거나 주로 쓴 곳이고, 5장 표시는 앞으로 나올 것입니다.

- [1부. `nn.`: 층과 모델](#1부-nn-층과-모델)
- [2부. `torch.`: 텐서 다루기](#2부-torch-텐서-다루기)
- [3부. 자세히 정리한 것](#3부-자세히-정리한-것)

---

## 1부. `nn.`: 층과 모델

### 뼈대

| 이름 | 하는 일 | 예시 | 어디서 |
|---|---|---|---|
| `nn.Module` | 모든 모델과 층의 부모 클래스. 상속하고 `__init__`, `forward`를 쓰면 내 층이 됨 | `class CausalAttention(nn.Module):` | 3·4장, Do it! |
| `nn.Parameter` | 텐서를 **학습할 값**으로 등록 | `nn.Parameter(torch.rand(3, 2))` | 3장 `SelfAttention_v1`, 4장 `LayerNorm`의 scale·shift |
| `self.register_buffer` | 텐서를 **학습하지 않는 값**으로 등록. 모델과 함께 GPU로 옮겨짐 | `register_buffer("mask", torch.triu(...))` | 3장 코잘 마스크 |

`register_buffer`는 `nn.` 함수가 아니라 `nn.Module`이 가진 메서드다.

### 층 (가중치가 있음)

| 이름 | 하는 일 | 모양 변화 | 어디서 |
|---|---|---|---|
| `nn.Linear(입력, 출력)` | 가중치를 곱하고 편향을 더함 | 마지막 차원: 입력 → 출력 | 3장 Q·K·V, `out_proj` / 4장 FeedForward, 출력층 |
| `nn.Embedding(어휘 수, 차원)` | 토큰 ID를 벡터로 바꾸는 표 | `(b, 토큰 수)` → `(b, 토큰 수, 차원)` | 2장 토큰·위치 임베딩 / 4장 GPTModel |
| `nn.LayerNorm(차원)` | 벡터마다 평균 0, 분산 1로 맞춤 | 바뀌지 않음 | 4장 (책은 직접 구현) |

### 활성화 함수 (가중치 없음)

| 이름 | 하는 일 | 특징 | 어디서 |
|---|---|---|---|
| `nn.ReLU()` | 음수는 0, 양수는 그대로 | 간단하고 빠름, 0에서 꺾임. 음수 쪽 기울기가 0이라 "죽은 ReLU" 문제 | 4장 비교용 |
| `nn.GELU()` | ReLU와 비슷하지만 부드러운 곡선 | 음수 쪽에도 작은 값이 남음 (GELU(-1) ≈ -0.16) | 4장 FeedForward (책은 직접 구현), BERT |

활성화 함수가 필요한 이유: `nn.Linear`만 여러 개 쌓으면 결국 `nn.Linear` 하나와 같다. 사이에 비선형 함수를 넣어야 층을 쌓은 의미가 생긴다.

### 학습을 돕는 것

| 이름 | 하는 일 | 주의 | 어디서 |
|---|---|---|---|
| `nn.Dropout(p)` | 학습할 때 값의 p 비율을 무작위로 0으로 만들고, 남은 값은 `1/(1-p)`배 키움 | `model.eval()`이면 꺼짐 | 3장 어텐션 가중치 / 4장 임베딩 뒤, 블록 안 |

### 층을 담는 그릇

| 이름 | 하는 일 | 호출 | 어디서 |
|---|---|---|---|
| `nn.Sequential(층1, 층2, ...)` | 담고 **순서대로 자동 실행** | `self.layers(x)` 한 줄 | 4장 FeedForward, 블록 12개 쌓기 |
| `nn.ModuleList([...])` | 담기**만** 함. 실행은 직접 | `for head in self.heads:` | 3장 `MultiHeadAttentionWrapper` |

- `nn.Sequential(*[TransformerBlock(cfg) for _ in range(12)])`: 앞의 `*`는 리스트를 풀어서 하나씩 넣어 준다.
- `nn.Sequential`은 한 줄로 이어지는 흐름만 표현할 수 있다. 숏컷 연결(`x = x + shortcut`)처럼 입력을 들고 있다가 더해야 하면 `forward`를 직접 쓴다.
- 3장 래퍼는 같은 `x`를 각 헤드에 **따로** 넣고 `torch.cat`으로 붙여야 해서 `nn.ModuleList`를 썼다.

### 손실 함수

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `nn.CrossEntropyLoss()` / `F.cross_entropy` | 정답 클래스의 확률이 높을수록 작아지는 손실. softmax까지 안에서 해 줌 | Do it! 분류·NER / 5장 사전학습 |

정답 위치에 `-100`을 넣으면 그 자리는 손실 계산에서 빠진다 (NER의 `[CLS]`, `[PAD]` 자리).

### `nn.` 과 `F.`

| | `nn.ReLU()` | `F.relu(x)` |
|---|---|---|
| 형태 | 층 객체 | 함수 |
| 쓰는 곳 | `__init__`에서 만들어 두기, `nn.Sequential`에 넣기 | `forward` 안에서 바로 부르기 |
| 결과 | 같음 | 같음 |

`F`는 `import torch.nn.functional as F`.

---

## 2부. `torch.`: 텐서 다루기

`torch.함수(x)`와 `x.함수()` 두 가지로 다 쓸 수 있는 것이 많다. 실제로 쓴 형태로 적었다.

### 텐서 만들기

| 이름 | 만드는 것 | 예시 | 어디서 |
|---|---|---|---|
| `torch.tensor(데이터)` | 리스트로 텐서 만들기 | `torch.tensor([[0.43, 0.15, 0.89], ...])` | 3장 `inputs`, NSMC 라벨 |
| `torch.zeros(모양)` | 전부 0 | `torch.zeros(6)` | 3장 `context_vec_2`, 4장 LayerNorm shift |
| `torch.ones(모양)` | 전부 1 | `torch.ones(6, 6)` | 3장 마스크 재료, 4장 LayerNorm scale |
| `torch.empty(모양)` | 값을 안 채운 빈 텐서 (곧 덮어쓸 때) | `torch.empty(6, 6)` | 3장 `attn_scores` for문 |
| `torch.arange(n)` | 0, 1, ..., n-1 | `torch.arange(context_length)` | 2·4장 위치 임베딩 |
| `torch.zeros_like(x)` / `torch.ones_like(x)` | x와 같은 모양으로 0/1 | `torch.ones_like(x)` | |

### 무작위

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `torch.manual_seed(123)` | 무작위 값을 매번 같게 고정 | 3·4장 거의 모든 셀 |
| `torch.rand(모양)` | 0~1 균등분포 | 3장 `SelfAttention_v1` |
| `torch.randn(모양)` | 평균 0, 표준편차 1 정규분포 | 4장 테스트 입력 `batch_example` |
| `torch.randint(낮, 높, 모양)` | 정수 무작위 | 가짜 토큰 ID |
| `torch.multinomial(확률, 1)` | 확률대로 하나 뽑기 | 5장 텍스트 생성 (샘플링) |

- `rand`는 0~1 양수만, `randn`은 0 근처에 몰린 값(약 68%가 -1~1)이 음수까지 나온다.
- 시드를 고정한 **뒤** 무작위 함수를 부르는 순서까지 책과 같아야 같은 숫자가 나온다. 층을 하나 더 만들거나 `randn`을 한 번 더 부르면 그다음 값이 전부 달라진다.

### 모양 바꾸기

| 이름 | 하는 일 | 예시 | 어디서 |
|---|---|---|---|
| `x.shape` | 모양 확인 | `b, num_tokens, d_in = x.shape` | 어디서나 |
| `x.view(모양)` | 순서 그대로 다시 끊기 | `(2, 6, 2) → (2, 6, 2, 1)` | 3장 멀티 헤드 |
| `x.reshape(모양)` | view와 같음, 필요하면 복사 | `logits.reshape(-1, num_labels)` | NER `compute_loss` |
| `x.transpose(i, j)` | i, j 차원 맞바꾸기 | `keys.transpose(1, 2)` | 3장 |
| `x.T` | 2차원 전치 (3차원 이상에는 쓰지 않기) | `inputs @ inputs.T` | 3장 3.3.2 |
| `x.contiguous()` | 메모리 순서 다시 정렬 (view 전에) | `.contiguous().view(...)` | 3장 |
| `x.unsqueeze(dim)` | 크기 1인 차원 추가 | `(4,) → (1, 4)` | 4장 생성 (배치 차원 붙이기) |
| `x.squeeze(dim)` | 크기 1인 차원 제거 | `(1, 4) → (4,)` | 4장 생성 결과 디코딩 |
| `x.flatten()` | 한 줄로 펴기 | `logits.flatten(0, 1)` | 5장 손실 계산 |

### 합치기·자르기

| 이름 | 하는 일 | 예시 | 어디서 |
|---|---|---|---|
| `torch.cat([a, b], dim)` | **있는** 차원으로 이어 붙이기 | `(2,6,2)+(2,6,2) → (2,6,4)` | 3장 래퍼, 4장 생성 |
| `torch.stack([a, b], dim)` | **새** 차원을 만들며 쌓기 | `(6,3)+(6,3) → (2,6,3)` | 2장 배치, 3장 `batch`, 4장 입력 |
| `x[:, -1, :]` | 인덱싱·슬라이싱 | 마지막 토큰만 | 4장 생성 |
| `x[:n, :n]` | 앞부분 자르기 | `self.mask[:num_tokens, :num_tokens]` | 3장 마스크 |

### 마스크·조건

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `torch.triu(x, diagonal=1)` | 대각선 위쪽만 남김 | 3장 코잘 마스크 |
| `torch.tril(x)` | 대각선 아래쪽만 남김 | 3장 3.5.1 첫 번째 방법 |
| `x.masked_fill_(조건, 값)` | 조건이 True인 칸을 값으로 채움 (제자리) | 3장 `-inf` 채우기 |
| `x.masked_fill(조건, 값)` | 위와 같은데 새 텐서를 돌려줌 | 결과를 받아 써야 함 |
| `x.bool()` | 0/1 → False/True | 3장 마스크 |
| `torch.where(조건, a, b)` | 조건에 따라 a나 b 고르기 | 5장 top-k 샘플링 |
| `torch.inf` | 무한대 | 3장 `-torch.inf` |

### 계산 (원소마다)

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `torch.softmax(x, dim=-1)` | 합이 1인 비율로 | 3장 어텐션 가중치, 4장 생성 |
| `torch.tanh(x)` | -1~1로 누르기 | 4장 GELU 근사식 |
| `torch.sqrt(x)` / `x ** 0.5` | 제곱근 | 3장 `keys.shape[-1] ** 0.5`, 4장 LayerNorm |
| `torch.pow(x, 3)` / `x ** 3` | 거듭제곱 | 4장 GELU의 x³ |
| `torch.exp(x)` | eˣ | 3장 `softmax_naive`, 5장 퍼플렉서티 |
| `torch.log(x)` | 자연로그 | 5장 손실 직접 계산 |

### 줄이기 (dim 방향으로 하나로)

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `x.sum(dim)` | 합 | 3장 "행마다 합이 1" 확인 |
| `x.mean(dim, keepdim=True)` | 평균 | 4장 LayerNorm, 손실 평균 |
| `x.var(dim, unbiased=False)` | 분산 (n으로 나눔) | 4장 LayerNorm |
| `x.argmax(dim=-1)` | 가장 큰 값의 **번호** | NSMC·NER 예측, 4장 생성 |
| `torch.topk(x, k)` | 가장 큰 k개 | 5장 top-k 샘플링 |
| `x.numel()` | 원소 개수 | 4장 파라미터 수 세기 |

### 행렬 계산

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `a @ b` / `torch.matmul(a, b)` | 행렬 곱 (마지막 두 차원) | 3장 어텐션 전체 |
| `torch.dot(a, b)` | 1차원 벡터 내적 | 3장 3.3.1 첫 점수 계산 |
| `torch.allclose(a, b, atol)` | 두 텐서가 거의 같은지 | 노트북 확인 셀 `_close` |
| `torch.equal(a, b)` | 완전히 같은지 | 4장 생성 확인 셀 |

### 장치·자료형

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `torch.cuda.is_available()` | GPU가 있는지 | NSMC·NER `device` 정하기 |
| `torch.device("cuda")` | 장치 지정 | NSMC·NER |
| `x.to(device)` / `model.to(device)` | GPU/CPU로 옮기기 | NSMC·NER 학습 루프 |
| `x.float()` / `x.long()` | 실수형 / 정수형으로 | 라벨은 `long`, 확인 셀은 `float` |
| `x.item()` | 값 하나짜리 텐서 → 파이썬 숫자 | `loss.item()` |
| `x.tolist()` | 텐서 → 파이썬 리스트 | 4장 생성 토큰 디코딩, NER 예측 |
| `x.detach()` | 기울기 추적 끊기 | 확인 셀, 그림 그릴 때 |
| `x.numpy()` | NumPy 배열로 | 그래프 그릴 때 |

### 학습 관련

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `loss.backward()` | 기울기 계산 | NSMC·NER, 4장 숏컷 실험 |
| `p.grad` | 계산된 기울기 | 4장 숏컷 실험 |
| `torch.no_grad()` | 기울기 계산 끄기 (평가·생성 때) | `evaluate`, 4장 생성 |
| `torch.optim.AdamW(...)` | 최적화 도구 | NSMC·NER, 5장 |
| `torch.autocast(...)` | 16비트로 빠르게 계산 (AMP) | NSMC·NER |
| `torch.amp.GradScaler()` | AMP에서 기울기 크기 조절 | NSMC·NER |
| `torch.nn.functional.cross_entropy` | 분류 손실 | NER, 5장 |
| `torch.save` / `torch.load` | 모델 저장·불러오기 | 5장 |

### 데이터

| 이름 | 하는 일 | 어디서 |
|---|---|---|
| `torch.utils.data.Dataset` | 데이터셋의 부모 클래스 (`__len__`, `__getitem__`) | NSMC, NER, 2장 `GPTDatasetV1` |
| `torch.utils.data.DataLoader` | 배치로 묶고 섞어서 꺼내 줌 | 어디서나 |

### 헷갈리기 쉬운 짝

| 짝 | 차이 |
|---|---|
| `view` vs `reshape` | reshape는 transpose 뒤에도 바로 됨 |
| `view` vs `transpose` | view는 순서 그대로 끊기, transpose는 축 맞바꾸기 |
| `cat` vs `stack` | cat은 있는 차원을 늘림, stack은 새 차원을 만듦 |
| `masked_fill` vs `masked_fill_` | `_`가 있으면 원래 텐서를 직접 고침 |
| `rand` vs `randn` | 0~1 균등 vs 평균 0 정규 |
| `triu` vs `tril` | 위쪽 남기기 vs 아래쪽 남기기 |
| `max` vs `argmax` | 가장 큰 **값** vs 가장 큰 값의 **번호** |
| `squeeze` vs `unsqueeze` | 크기 1인 차원 빼기 vs 넣기 |
| `ModuleList` vs `Sequential` | 담기만 함 vs 담고 순서대로 실행 |

---

## 3부. 자세히 정리한 것

### `nn.Linear`

```
nn.Linear(입력, 출력)
= 학습 가능한 W (출력 × 입력) + b (출력)
= 출력 한 칸 = 입력 전체의 가중합 + 편향
= 마지막 차원만 바꾸고, 앞 차원(배치·토큰)에는 같은 W를 반복 적용
```

- 계산: `y = x @ W.T + b`. `weight`가 `(출력, 입력)` 순서로 저장되어 있어서 `.T`가 붙는다.
- W의 한 행 = 출력 한 칸을 만드는 가중치.
  ```
  x = [1, 2, 3],  W = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  b = [0.01, 0.02]
  y1 = 0.1·1 + 0.2·2 + 0.3·3 + 0.01 = 1.41
  y2 = 0.4·1 + 0.5·2 + 0.6·3 + 0.02 = 3.22
  ```
- 직접 확인: `layer(x)`와 `x @ layer.weight.T + layer.bias`는 같은 값.
- 모양: `(3,) → (2,)`, `(6, 3) → (6, 2)`, `(2, 6, 3) → (2, 6, 2)`. 문장 길이가 달라도 같은 층을 쓸 수 있다.
- 파라미터 수 = 입력 × 출력 + 출력. 셀 때는 `sum(p.numel() for p in layer.parameters())`.

  | 층 | 파라미터 수 |
  |---|---|
  | `nn.Linear(768, 768, bias=False)` (W_query) | 589,824 |
  | `nn.Linear(768, 3072)` (FeedForward 1) | 2,362,368 |
  | `nn.Linear(768, 50257, bias=False)` (출력층) | 38,597,376 |

  4장에서 파라미터를 세면 1억 2,400만이 아니라 1억 6,300만이 나온다. GPT-2는 출력층을 토큰 임베딩과 **공유(weight tying)**해서 출력층을 따로 세지 않기 때문.
- `bias=False`: 편향 없이 곱하기만. Q·K·V는 점곱으로 비교하는 데만 쓰여서, 그리고 LayerNorm 바로 앞에서는 평균을 빼면서 편향이 지워져서 보통 뺀다.
- 초기값: `-1/√입력 ~ +1/√입력` 범위에서 균등하게 뽑는다. 입력이 많을수록 범위를 좁혀 처음부터 출력이 커지지 않게 한다. 3장 `SelfAttention_v1`(`torch.rand`, 0~1)과 v2(`nn.Linear`)의 출력이 다른 이유.
- 학습: `loss.backward()`로 `W.grad`, `b.grad`가 채워지고 `optimizer.step()`이 `W = W - 학습률 × W.grad`처럼 고친다.

### `.view()`와 `.contiguous()`

- `.view()`는 데이터를 복사하지 않고 메모리에 놓인 순서대로 다시 끊어 읽기만 한다. 전체 개수가 같아야 하고, `-1`은 "나머지는 알아서"(한 번만).
- `.transpose()`는 데이터를 옮기지 않고 읽는 방향만 바꾼다. 그래서 transpose 직후에 `.view()`를 하면 `view size is not compatible with input tensor's size and stride` 에러가 난다.
- `.contiguous()`가 보이는 모양 순서대로 메모리에 다시 복사해 정렬한다. 다음 두 줄은 결과가 같다.
  ```python
  x.transpose(1, 2).contiguous().view(...)
  x.transpose(1, 2).reshape(...)
  ```
- view와 transpose는 모양이 같아도 내용 배치가 다르다.
  ```python
  x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
  x.view(3, 2)        # [[1, 2], [3, 4], [5, 6]]
  x.transpose(0, 1)   # [[1, 4], [2, 5], [3, 6]]
  ```

### `dim=-1`

- 마지막 차원. `(2, 6, 3)`에서 앞에서부터 `0, 1, 2`, 뒤에서부터 `-3, -2, -1`.
- 차원 수가 달라도 항상 마지막을 가리켜서 `(6, 6)`, `(2, 6, 6)`, `(2, 2, 6, 6)` 점수에 모두 `torch.softmax(attn_scores, dim=-1)` 한 줄로 쓸 수 있다.
- "dim 방향으로 계산" = 그 차원을 따라 움직이며 계산하고 그 차원이 사라진다.
  ```python
  x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
  x.sum(dim=0)    # [5., 7., 9.]   위아래로 (열마다)
  x.sum(dim=-1)   # [6., 15.]      옆으로 (행마다)
  ```
- softmax에서 dim을 틀리면 에러 없이 엉뚱한 방향으로 정규화된다. 그래서 확인 셀에서 "각 행의 합이 1"을 검사한다.
- `keepdim=True`는 사라질 차원을 크기 1로 남긴다. `(2, 5) - (2, 1)`처럼 브로드캐스팅하려면 필요 (4장 LayerNorm).

### `torch.triu`의 `diagonal`

- 대각선 번호 = 열 번호 - 행 번호. `triu(..., diagonal=k)`는 번호가 k 이상인 칸만 남긴다.
  ```
  diagonal=0          diagonal=1          diagonal=-1
  [1 1 1 1]           [0 1 1 1]           [1 1 1 1]
  [0 1 1 1]           [0 0 1 1]           [1 1 1 1]
  [0 0 1 1]           [0 0 0 1]           [0 1 1 1]
  [0 0 0 1]           [0 0 0 0]           [0 0 1 1]
  ```
- 코잘 마스크(1 = 가릴 칸)는 미래 단어만 가려야 해서 `diagonal=1`. `0`이면 자기 자신까지, `-1`이면 바로 앞 단어까지 가려서 첫 행이 전부 `-inf` → softmax가 `nan`.
- `tril`은 반대로 대각선과 아래쪽을 남긴다 (1 = 볼 수 있는 칸, softmax 뒤에 곱하고 다시 정규화).
