# 3장 어텐션 메커니즘 구현하기: 정리

> 책 문장을 옮기지 말고 내 말로 적기. 다 적은 뒤 “이걸 처음 보는 친구에게 설명할 수 있나?”로 점검하기
> 

## 한 줄 요약

- 3장에서 배운 것:

## 핵심 개념

| 개념 | 내 말로 설명 |
| --- | --- |
| RNN의 한계 (3.1) | 문장이 길어질수록  인코더 앞부분의 내용(그레디언트)를 잃을 가능성이 높다 |
| 어텐션 점수 | 어텐션 점수는 queries @ keys.T (keys의 전치, 역행렬 아님. 배치가 있으면 keys.transpose(1, 2))로 보통 표현되며, 원소와 다른원소 사이의 점곱이다 |
| 어텐션 가중치 | 어텐션 점수/ 키의 차원수 **0.5에 소프트맥스를 한 것으로 표현된다.  |
| 컨텍스트 벡터 | 어텐션가중치@values로 표현되며, 값들을 가중치에 곱해 최종적인 문맥벡터를 표현한다 |
| 쿼리 · 키 · 밸류 | query는 현재 초점을 맞추거나 이해할 항목, key는 입력 시퀀스에 있는 각 항목의 key, value는 실제 콘텐츠/표현. 즉 input이 key-value 쌍으로 이루어져 있을 때 query가 특정 집중된 부분을 나타낸다 |
| √d_k로 나누는 이유 | 키의 차원수의 제곱근으로 나누는 이유는 점곱의 결과가 커질 때 그레디언트가 0에 가까워지는 것을 막기 위해서이다 |
| 코잘 마스크 | 특정 부분에 0또는 -inf로 마스킹을 하여 모델이 이전 또는 현재 부분만 참고하여 학습하도록 한다 |
| 드롭아웃 (어텐션에서) | 과적합을 방지하기 위해 랜덤하게 해당 유닛의 값을 무시한다 |
| 멀티 헤드 어텐션 | 싱글 어텐션을 여러개 겹쳐 더 복잡한 문장도 처리할 수 있게 한다 |

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
.view(b,num_tokens,self.d_out)=(2,6,2) |
| `out_proj` 뒤 최종 출력 | (2, 6, 2) |

## Do it! 공부 노트와 비교

|  | Do it! 노트에서 정리한 것 | 3장에서 직접 만들어 보니 |
| --- | --- | --- |
| Q, K, V | Q = XW_Q, K = XW_K, V = XW_V |  |
| 어텐션 공식 | softmax(QK^T / √d_k) V | 이 공식 안에 어텐션 점수, 어텐션 가중치, 최종적으로 문맥벡터를 구하는 과정이 있었다 |
| 디코더 셀프 어텐션 | 뒤 단어를 마스킹 | 마스킹의 과정에서 -inf를 쓰면 더 효율적으로 계산할 수 있다 |
| 멀티 헤드 | 여러 헤드 출력을 이어 붙인 뒤 W_O를 곱함 | 이어붙일 때 출력 투영 층의 존재가 필요하다. 그저 이어붙이기만 하면 정보가 섞이지 않고 자기 헤드 하나의 의견만 가짐⇒ out_proj로 입력의 모든 칸을 가중합 |
| BERT vs GPT | BERT는 앞뒤를 다 봄, GPT는 앞만 봄 | BERT와 GPT를 과제에 맞게 쓰는 것이 중요하다 |

## 다시 짜기 체크

- [x]  `CausalAttention`을 보지 않고 다시 짜기

    ```python
    class CausalAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            values = self.W_value(x)
            queries = self.W_query(x)
            attn_scores = queries @ keys.transpose(1, 2)  # 배치 축은 두고 마지막 두 축만 바꿈 (3차원에 .T를 쓰면 축이 전부 뒤집힘)

            attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = attn_weights @ values
            return context_vec
    ```

- [x]  `MultiHeadAttention`을 보지 않고 다시 짜기 (`attention.py`로 저장)

    ```python
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.num_heads = num_heads  # forward에서 쓰므로 저장
            self.head_dim = d_out // num_heads

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)  # 헤드의 출력을 결합할 Linear 층 (입력은 이어 붙인 헤드 출력 → d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape  # 텐서크기를 각각의 변수에 담기

            keys = self.W_key(x)  # 크기⇒b,num_tokens,d_out
            queries = self.W_query(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # d_out의 차원을 num_heads와 head_dim 두개의 차원으로 다시 나눔
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            keys = keys.transpose(1, 2)  # (b,num_heads,num_tokens, head_dim)차원으로 변환
            values = values.transpose(1, 2)
            queries = queries.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)  # 행렬의 차원이 4개로 늘어 마지막 2개 차원인 2,3을 변환 (≠keys.T)

            attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # 논리형 마스크 호출(1,0으로 이루어진 마스크)⇒토큰 갯수로 자르기⇒1(True, 미래 토큰) 자리를 -inf로 채우기. 밑줄 있는 masked_fill_이라야 attn_scores가 직접 바뀜

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)  # 다시 원래 형태로 되돌리기(b,num_tokens,num_heads,head_dim)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # self.num_heads와head_dim을 곱해서 다시 원래 형태로 되돌림
            context_vec = self.out_proj(context_vec)
            return context_vec
    ```


## 이해 확인 질문

1. 어텐션 점수를 그대로 쓰지 않고 softmax로 정규화하는 이유는?
    
    어텐션 점수는 합해서 1이 되지 않아서 가중치로 사용할 수 없다
    
2. √d_k로 나누지 않으면 어떤 문제가 생길까?
    
    점곱 값(어텐션 점수)이 너무 커져서 softmax가 한 칸에 거의 1을 몰아주고 나머지 가중치는 0에 가까워진다 = 그레디언트가 0에 가까워져 학습이 잘 안 된다
    
3. 코잘 마스크에서 softmax 뒤에 0을 곱하는 대신, softmax 앞에 `inf`를 채우는 이유는?
    
    -inf는 softmax에서 0으로 처리되기 때문에 이후에 다른 연산을 할 필요가 없다
    
4. 드롭아웃은 왜 학습할 때만 쓰고 예측할 때는 끌까?
    
    학습할 때는 과적합을 방지하기 위해 켜고, 예측할 땐 정확한 예측을 위해서 모든 데이터를 적용한다
    
5. `MultiHeadAttentionWrapper`와 `MultiHeadAttention`은 결과는 같은 역할인데 무엇이 다를까? 왜 뒤의 것이 더 효율적일까?
    
    Wrapper는 단순히 여러번 연산을 한 결과를 합치는 것이고, multiheadattention은 행렬의 차원을 나누어 한번에 연산을 수행한 뒤 곱하여 연산이 훨씬 줄어든다
    

## 궁금한 점·더 알아볼 것

-