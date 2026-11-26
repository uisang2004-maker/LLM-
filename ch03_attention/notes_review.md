# 3장 정리 노트 첨삭 (notes.md 검토)

> **첨삭 안내**: 원래 노트 문장은 그대로 두고, 고칠 점·보탤 점만 `✏️ 첨삭` 블록으로 덧붙였습니다. 규칙("내 말로 적기")대로 첨삭은 힌트로만 쓰고, 노트에는 직접 다시 써 보세요.
> 표시: ✅ 맞음 / ⚠️ 보완하면 좋음 / ❌ 틀림
> 코드 수정본은 같은 폴더의 `attention.py`에 있습니다(고친 줄마다 `# [수정]` 주석).

# 3장 어텐션 메커니즘 구현하기: 정리

> 책 문장을 옮기지 말고 내 말로 적기. 다 적은 뒤 “이걸 처음 보는 친구에게 설명할 수 있나?”로 점검하기
> 

## 한 줄 요약

- 3장에서 배운 것:

> ✏️ 첨삭 ⚠️ 비어 있음. 아래 표를 다 채웠으니 한 문장으로 묶어 보세요. (힌트: 무엇을 → 어떻게 계산해서 → 무엇을 만드는지)

## 핵심 개념

| 개념 | 내 말로 설명 |
| --- | --- |
| RNN의 한계 (3.1) | 문장이 길어질수록  앞인코더의 내용(그레디언트)를 잃을 가능성이 높다 |
| 어텐션 점수 | 어텐션 점수는 queries @ key.T 로 보통 표현되며, 원소와 다른원소 사이의 점곱이다 |
| 어텐션 가중치 | 어텐션 점수/ 키의 차원수 **0.5에 소프트맥스를 한 것으로 표현된다.  |
| 컨텍스트 벡터 | 어텐션가중치@values로 표현되며, 값들을 가중치에 곱해 최종적인 문맥벡터를 표현한다 |
| 쿼리 · 키 · 밸류 | query는 현재 초점을 맞추거나 이해할 항목, key는 입력 시퀀스에 있는 각 항목의 key, value는 실제 콘텐츠/표현. 즉 input이 key-value 쌍으로 이루어져 있을 때 query가 특정 집중된 부분을 나타낸다 |
| √d_k로 나누는 이유 | 키의 차원수의 제곱근으로 나누는 이유는 점곱의 결과가 커질 때 그레디언트가 0에 가까워지는 것을 막기 위해서이다 |
| 코잘 마스크 | 특정 부분에 0또는 -inf로 마스킹을 하여 모델이 이전 또는 현재 부분만 참고하여 학습하도록 한다 |
| 드롭아웃 (어텐션에서) | 과적합을 방지하기 위해 랜덤하게 해당 유닛의 값을 무시한다 |
| 멀티 헤드 어텐션 | 싱글 어텐션을 여러개 겹쳐 더 복잡한 문장도 처리할 수 있게 한다 |

> ✏️ 첨삭 — 핵심 개념
> - **RNN의 한계** ⚠️ 방향은 맞음. 더 정확히는: 인코더가 입력 전체를 **마지막 은닉 상태 하나**에 몰아 담아야 해서 문장이 길면 앞부분 정보가 희미해짐. "그레디언트"는 학습할 때 생기는 문제(기울기 소실)라 "내용"과는 구분해서 쓰면 좋음. "앞인코더"는 "인코더 앞부분"의 오타로 보임.
> - **어텐션 점수** ⚠️ `key.T`는 `keys.T`(이름 오타). 그리고 `.T`는 **전치(transpose)**이지 역행렬이 아님. 배치가 있는 3차원 텐서에서는 `keys.transpose(1, 2)`, 멀티헤드 4차원에서는 `keys.transpose(2, 3)`처럼 **마지막 두 축만** 바꿔야 함. "원소와 다른원소" → "각 쿼리와 각 키 사이의 점곱"이라고 쓰면 더 분명함.
> - **어텐션 가중치** ✅ 식은 맞음: softmax(점수 / √d_k). 덧붙이면 좋은 것: 결과의 각 행 합이 1이 되어 "얼마나 볼지"를 비율로 나타냄.
> - **컨텍스트 벡터** ✅
> - **쿼리·키·밸류** ✅
> - **√d_k로 나누는 이유** ✅ 거의 정확함. 중간 단계를 넣으면 완벽: 차원이 클수록 점곱 값이 커짐 → softmax가 한 칸에 거의 1을 몰아주는(뾰족한) 분포가 됨 → 그 상태에서는 기울기가 0에 가까워져 학습이 잘 안 됨.
> - **코잘 마스크** ⚠️ "0 또는 -inf" 두 방법의 차이를 적어 두면 좋음(아래 이해 확인 3번 참고). 또 **어디를** 가리는지: 대각선 **위쪽** = 현재 토큰보다 **뒤(미래)** 토큰.
> - **드롭아웃** ⚠️ 어텐션에서는 **어텐션 가중치 행렬**에 적용한다는 점을 넣으면 좋음. 그리고 학습 때만 켬(이해 확인 4번).
> - **멀티 헤드** ⚠️ "여러 개 겹쳐"보다는: d_out을 여러 헤드로 **나눠서** 각 헤드가 서로 다른 관계(패턴)를 보도록 하고, 마지막에 이어 붙여 out_proj로 섞음.

## 텐서 모양 따라가기

`MultiHeadAttention`에서 `batch`(2, 6, 3)가 줄마다 어떤 모양이 되는지 적어 보기 (`d_out=2`, `num_heads=2`)

| 단계 | 모양 |
| --- | --- |
| 입력 `x` | (2, 6, 3) #배치, 토큰수, 벡터길이 |
| `keys = self.W_key(x)` | (2,6,2) #W_key는 (d_in,d_out)⇒ (3,2)⇒ 마지막 줄이 2로 바뀐다 |
| `keys.view(b, num_tokens, num_heads, head_dim)` | (2,6,2,1)#배치크기,토큰수, 헤드수, 헤드당 차원수 |
| `.transpose(1, 2)`  | (2,2,6,1)#1번자리와 2번자리 변환 |
| `queries @ keys.transpose(2, 3)` | (2,2,6,1)@(2,2,1,6)= (2,2,6,6) |
| 컨텍스트 벡터를 다시 합친 뒤 | attn_weights @ values = (2,2,6,1)
.transpose(1,2)= (2,6,2,1)
.view(b,bum_tokens,self.d_out)=(2,6,2) |
| `out_proj` 뒤 최종 출력 | (2, 6, 2) |

> ✏️ 첨삭 — 텐서 모양 ✅ 모양은 **전부 맞음**. `attention.py`를 실제로 실행해서 확인함(아래 "실행 확인" 참고).
> - `bum_tokens` → `num_tokens` (오타)
> - 작은 보충: `nn.Linear(3, 2)`의 가중치는 PyTorch 안에서 (2, 3)으로 저장됨(출력, 입력 순). "(d_in, d_out) ⇒ 마지막 차원이 2로 바뀐다"는 이해는 맞음.
> - `.view()` 앞에 `.contiguous()`가 필요한 이유도 한 줄 적어 두면 좋음: transpose 뒤에는 메모리 배치가 연속이 아니라서 view가 안 될 수 있음.

## Do it! 공부 노트와 비교

|  | Do it! 노트에서 정리한 것 | 3장에서 직접 만들어 보니 |
| --- | --- | --- |
| Q, K, V | Q = XW_Q, K = XW_K, V = XW_V |  |
| 어텐션 공식 | softmax(QK^T / √d_k) V | 이 공식 안에 어텐션 점수, 어텐션 가중치, 최종적으로 문맥벡터를 구하는 과정이 있었다 |
| 디코더 셀프 어텐션 | 뒤 단어를 마스킹 | 마스킹의 과정에서 -inf를 쓰면 더 효율적으로 계산할 수 있다 |
| 멀티 헤드 | 여러 헤드 출력을 이어 붙인 뒤 W_O를 곱함 | 이어붙일 때 출력 투영 층의 존재가 필요하다. 그저 이어붙이기만 하면 정보가 섞이지 않고 자기 헤드 하나의 의견만 가짐⇒ out_proj로 입력의 모든 칸을 가중합 |
| BERT vs GPT | BERT는 앞뒤를 다 봄, GPT는 앞만 봄 | BERT와 GPT를 과제에 맞게 쓰는 것이 중요하다 |

> ✏️ 첨삭 — Do it! 비교
> - **Q, K, V** ⚠️ 오른쪽 칸이 비어 있음. 코드에서는 `nn.Linear(d_in, d_out, bias=False)` 세 개가 W_Q, W_K, W_V 역할을 한다는 점을 적어 보면 좋음.
> - **어텐션 공식** ✅ 좋은 정리. 공식의 `QK^T`에서 `^T`가 전치라는 것도 같이 적어 두면 코드의 `transpose`와 바로 연결됨.
> - **디코더 셀프 어텐션** ⚠️ "효율적"보다 정확한 이유: softmax 뒤에 0을 곱하면 각 행의 합이 1이 아니게 되어 **다시 정규화**해야 함. softmax 앞에 -inf를 넣으면 그 자리가 정확히 0이 되면서 나머지 합이 그대로 1이라 재정규화가 필요 없음.
> - **멀티 헤드** ✅ out_proj 설명이 좋음.
> - **BERT vs GPT** ⚠️ 3장 내용과 연결하면: GPT가 "앞만 보는" 이유가 바로 **코잘 마스크**.

## 다시 짜기 체크

> ✏️ 첨삭 — 두 코드 모두 구조와 흐름은 맞게 기억함. 실행하면 에러가 나는 곳만 고쳤고, 고친 코드는 `attention.py`에 있음.
>
> **CausalAttention** (고칠 곳 4개)
> 1. `def __init __` → `def __init__` (공백이 있으면 생성자로 인식되지 않음)
> 2. `’mask’` → `"mask"` (굽은 따옴표는 파이썬 문법 오류. 노션에서 자동으로 바뀌는 경우가 많음)
> 3. ❌ `queries @ key.T` → `queries @ keys.transpose(1, 2)`
>    - `key`는 없는 이름(`keys`가 맞음)
>    - 3차원 텐서에 `.T`를 쓰면 **모든 축이 뒤집혀** (d_out, num_tokens, b)가 됨. 배치 축은 두고 마지막 두 축만 바꿔야 함
> 4. `[ :num tokens, ...]` → `[:num_tokens, ...]` (오타)
>
> **MultiHeadAttention** (고칠 곳 10개)
> 1. `__init __`, `super().__init __()` → `__init__` (공백 2곳)
> 2. ❌ `self.num_heads` 저장 안 함 → `__init__`에 `self.num_heads = num_heads` 추가 (forward에서 씀)
> 3. `self.W_values` → `self.W_value` (forward에서 부르는 이름과 통일)
> 4. ❌ `self.out_proj = nn.Linear(d_in, d_out)` → `nn.Linear(d_out, d_out)`: out_proj의 입력은 x가 아니라 헤드를 이어 붙인 context_vec(마지막 차원 d_out). 노트 예시처럼 d_in=3, d_out=2이면 여기서 모양 에러가 남
> 5. `register_buffer(...)` 닫는 괄호 하나 빠짐 + 굽은 따옴표
> 6. ❌ `W_key(x)` 등 → `self.W_key(x)` (클래스 안의 층은 `self.`로 불러야 함)
> 7. ❌ `attn_scores.masked_fill(...)` → `masked_fill_(...)`: 밑줄 없는 버전은 **새 텐서를 돌려주기만** 하고 원본은 안 바뀜. 결과를 안 받으면 마스킹이 아예 적용되지 않음(에러도 안 나서 찾기 어려운 버그)
> 8. `self.mask.bool([:num_tokens, :num_tokens])` → `self.mask.bool()[:num_tokens, :num_tokens]` (괄호 위치)
> 9. `contigious` → `contiguous` (오타)
> 10. `.view(b, num_tokens, d_out)` → `self.d_out` (forward 안에는 `d_out` 변수가 없음)
>
> ✅ 주석은 아주 잘 달았음: view로 d_out을 (num_heads, head_dim)으로 나누는 이유, transpose(2,3)이 `keys.T`와 다른 이유, 마스크 처리 순서가 모두 정확함. (참고: `keys.transpose(-2, -1)`로 쓰면 축 개수와 상관없이 항상 마지막 두 축을 바꿈)
>
> ⚠️ 체크박스: `MultiHeadAttention` 항목이 `[ ]`로 되어 있음. `attention.py`로 저장까지 했으니 원본 노트에서 체크하세요.

- [x]  `CausalAttention`을 보지 않고 다시 짜기
    
    class CausalAttention(nn.Module):
    
    def __init __(self, d_in,d_out,context_length,dropout,qkv_bias=False):
    
    super().__init__()
    
    self.d_out=d_out
    
    self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
    
    self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
    
    self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)
    
    self.dropout=nn.Dropout(dropout)
    
    self.register_buffer(’mask’,torch.triu(torch.ones(context_length,context_length),diagonal=1))
    
    def forward(self,x):
    
    b, num_tokens, d_in= x.shape
    
    keys=self.W_key(x)
    
    values=self.W_value(x)
    
    queries=self.W_query(x)
    attn_scores= queries @ key.T
    
    attn_scores.masked_fill_( self.mask.bool()[ :num tokens, :num_tokens],-torch.inf)
    
    attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
    attn_weights=self.dropout(attn_weights)
    
    context_vec= attn_weights@values
    
    return context_vec
    
- [ ]  `MultiHeadAttention`을 보지 않고 다시 짜기 (`attention.py`로 저장)
    
    class MultiHeadAttention(nn.Module):
    
    def __init __(self, d_in, d_out, context_length, dropout,num_heads,qkv_bias=False):
    
    super().__init __()
    
    self.d_out=d_out
    
    self.head_dim=d_out//num_heads
    
    self.W_query=nn.Linear(d_in,d_out,qkv_bias)
    
    self.W_key=nn.Linear(d_in,d_out,qkv_bias)
    
    self.W_values=nn.Linear(d_in,d_out,qkv_bias)
    
    self.out_proj=nn.Linear(d_in,d_out) #헤드의 출력을 결합할 Linear 변수
    
    self.dropout=nn.Dropout(dropout)
    
    self.register_buffer(’mask’,torch.triu(torch.ones(context_length,context_length),diagonal=1)
    
    def forward(self,x):
    
    b,num_tokens,d_in=x.shape #텐서크기를 각각의 변수에 담기
    
    keys=W_key(x) #크기⇒b,num_tokens,d_out
    
    queries=W_query(x)
    
    values=W_value(x)
    
    keys=keys.view(b,num_tokens,self.num_heads,self.head_dim) #d_out의 차원을 num_heads와 head_dim 두개의 차원으로 다시 나눔
    
    values=values.view(b,num_tokens,self.num_heads,self.head_dim)
    
    queries=queries.view(b,num_tokens,self.num_heads,self.head_dim)
    
    keys=keys.transpose(1,2) #(b,num_heads,num_tokens, head_dim)차원으로 변환
    
    values=values.transpose(1,2)
    
    queries=queries.transpose(1,2) 
    
    attn_scores=queries@keys.transpose(2,3) #행렬의 차원이 4개로 늘어 마지막 2개 차원인 2,3을 변환 (≠keys.T)
    
    attn_scores.masked_fill(self.mask.bool([:num_tokens, :num_tokens],-torch.inf) #논리형 마스크 호출(1,0으로 이루어진 마스크)⇒토큰 갯수로 자르기⇒0은 -inf로 채우기
    
    attn_weights=torch.softmax( attn_scores/keys.shape[-1]**0.5,dim=-1)
    
    attn_weights=self.dropout(attn_weights)
    
    context_vec=(attn_weights@values).transpose(1,2)#다시 원래 형태로 되돌리기(b,num_tokens,num_heads,head_dim)
    
    context_vec=context_vec.contigious().view(b,num_tokens,d_out) #self.num_heads와head_dim을 곱해서 다시 원래 형태로 되돌림
    
    context_vec=self.out_proj(context_vec)
    
    return context_vec

> ✏️ 첨삭 — 마스크 주석의 "0은 -inf로 채우기" ⚠️ 반대임. `torch.triu(..., diagonal=1)`은 대각선 **위쪽이 1**인 행렬이고, `.bool()`로 바꾸면 그 1(True) 자리 = **미래 토큰** 자리가 -inf로 채워짐. 0인 자리(자기 자신과 앞 토큰)는 그대로 남음.

## 이해 확인 질문

1. 어텐션 점수를 그대로 쓰지 않고 softmax로 정규화하는 이유는?
    
    어텐션 점수는 합해서 1이 되지 않아서 가중치로 사용할 수 없다

> ✏️ 첨삭 ✅ 맞음. 보태면: softmax는 음수 점수도 양수로 바꿔 줌 → "어디를 얼마나 볼지"를 0~1 사이 비율로 해석할 수 있음.

2. √d_k로 나누지 않으면 어떤 문제가 생길까?
    
    나누는 값이 너무 커져 가중치가 너무 작아진다 = 계산 효율이 떨어진다

> ✏️ 첨삭 ❌ 원인과 결과가 조금 엇갈림. 위 핵심 개념 표에 쓴 설명이 오히려 정답에 가까움.
> - 원인: 나누지 않으면 **점곱 값(점수)**이 차원 수에 따라 너무 커짐 ("나누는 값"이 커지는 게 아님)
> - 결과: softmax가 한 칸에 거의 1, 나머지는 거의 0인 뾰족한 분포가 됨
> - 문제: 그 상태에서는 **기울기가 0에 가까워져 학습이 잘 안 됨** (속도·효율 문제가 아니라 학습 안정성 문제)

3. 코잘 마스크에서 softmax 뒤에 0을 곱하는 대신, softmax 앞에 `inf`를 채우는 이유는?
    
    -inf는 softmax에서 0으로 처리되기 때문에 이후에 다른 연산을 할 필요가 없다

> ✏️ 첨삭 ⚠️ 거의 맞음. "다른 연산"이 무엇인지 적으면 완벽: softmax **뒤에** 0을 곱하면 남은 값들의 합이 1보다 작아져서 **각 행을 다시 나눠 정규화**해야 함. softmax **앞에** -inf를 넣으면 그 자리가 정확히 0이 되고 나머지 합이 그대로 1이라 재정규화가 필요 없음. (질문의 `inf`는 `-inf`가 맞음)

4. 드롭아웃은 왜 학습할 때만 쓰고 예측할 때는 끌까?
    
    학습할 때는 과적합을 방지하기 위해 끄고, 예측할 땐 정확한 예측을 위해서 모든 데이터를 적용한다

> ✏️ 첨삭 ❌ "학습할 때는 … 끄고" → **켜고**. (학습 때 켜고, 예측 때 끔) 이유 설명은 맞음.
> - 보충: 예측 때 끄는 이유는 매번 같은 입력에 같은 결과가 나와야 하기 때문(무작위성 제거). PyTorch에서는 `model.eval()`이 드롭아웃을 끄고, `model.train()`이 다시 켬. 학습 때 살아남은 값은 1/(1-p)배로 키워서 예측 때와 크기를 맞춤.

5. `MultiHeadAttentionWrapper`와 `MultiHeadAttention`은 결과는 같은 역할인데 무엇이 다를까? 왜 뒤의 것이 더 효율적일까?
    
    Wrapper는 단순히 여러번 연산을 한 결과를 합치는 것이고, multiheadattention은 행렬의 차원을 나누어 한번에 연산을 수행한 뒤 곱하여 연산이 훨씬 줄어든다

> ✏️ 첨삭 ⚠️ 방향은 맞음. 정확히 하면: 곱셈의 총량이 크게 줄어드는 게 아니라, Wrapper는 헤드마다 **따로** 행렬곱을 반복(for문)하고, MultiHeadAttention은 W_query 등 **큰 행렬곱 한 번**으로 모든 헤드를 동시에 계산한 뒤 view로 나눔 → GPU가 한꺼번에 병렬 처리할 수 있어 빠름. 또 MultiHeadAttention에는 헤드를 섞는 out_proj가 있다는 차이도 있음.

## 궁금한 점·더 알아볼 것

-

> ✏️ 첨삭 — 오늘 물어본 것을 여기 적어 두면 좋음
> - "attn_scores는 쿼리@키 **역행렬**인가?" → 아님. **전치**($K^\top$). 각 쿼리와 각 키의 점곱 표를 만들려고 행과 열을 바꾸는 것.
> - "왜 멀티헤드에서는 `transpose(2, 3)`인가?" → 텐서가 (b, num_heads, num_tokens, head_dim) 4차원이라 마지막 두 축 번호가 2, 3이기 때문. `@`는 마지막 두 축으로만 행렬곱을 하고 앞의 (b, num_heads)는 배치처럼 그대로 둠.

## 실행 확인 (attention.py)

> ✏️ 첨삭 — `attention.py`를 그대로 실행한 결과 (torch 2.15.0, CPU)
>
> ```
> batch.shape: (2, 6, 3)
> CausalAttention 출력: (2, 6, 2)
> MultiHeadAttention 출력: (2, 6, 2)
> 헤드 0의 어텐션 가중치 (윗삼각이 0이어야 함):
> [[1.    0.    0.    0.    0.    0.   ]
>  [0.486 0.514 0.    0.    0.    0.   ]
>  [0.321 0.339 0.339 0.    0.    0.   ]
>  [0.242 0.251 0.25  0.257 0.    0.   ]
>  [0.201 0.2   0.2   0.2   0.2   0.   ]
>  [0.156 0.166 0.166 0.173 0.164 0.175]]
> 각 행의 합: [1. 1. 1. 1. 1. 1.]
> ```
>
> - 노트의 "텐서 모양 따라가기" 표대로 입력 (2, 6, 3) → 최종 출력 (2, 6, 2) 확인 ✅
> - 대각선 위(미래 토큰)가 모두 0, 각 행의 합이 1 → 코잘 마스크와 softmax가 제대로 동작 ✅
> - 첫 토큰은 자기 자신만 볼 수 있어서 첫 행이 [1, 0, 0, 0, 0, 0]
