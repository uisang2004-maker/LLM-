"""3장 어텐션 메커니즘: CausalAttention, MultiHeadAttention

노트("다시 짜기 체크")의 코드를 실행되도록 고친 버전.
고친 곳마다 `# [수정]` 주석으로 무엇을 왜 바꿨는지 적어 둠.
맨 아래 `if __name__ == "__main__":` 부분은 노트의 "텐서 모양 따라가기" 표를 실제로 확인하는 테스트.
"""

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):  # [수정] __init __ → __init__ (공백 있으면 생성자로 인식 안 됨)
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",  # [수정] ’mask’ (굽은 따옴표) → "mask"
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)        # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # [수정] queries @ key.T → queries @ keys.transpose(1, 2)
        #   - key 는 정의되지 않은 이름(keys 가 맞음)
        #   - 3차원 텐서에 .T 를 쓰면 모든 축이 뒤집혀 (d_out, num_tokens, b) 가 됨.
        #     배치 축은 그대로 두고 마지막 두 축(토큰, 차원)만 바꿔야 함
        attn_scores = queries @ keys.transpose(1, 2)  # (b, num_tokens, num_tokens)

        # [수정] num tokens → num_tokens (오타)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):  # [수정] __init __ → __init__
        super().__init__()  # [수정] super().__init __() → super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어떨어져야 함"  # [추가] head_dim이 정수가 되도록 확인

        self.d_out = d_out
        self.num_heads = num_heads  # [추가] forward에서 self.num_heads를 쓰는데 저장을 안 했음
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # [수정] W_values → W_value (forward에서 쓰는 이름과 통일)
        # [수정] nn.Linear(d_in, d_out) → nn.Linear(d_out, d_out)
        #   out_proj의 입력은 x가 아니라 헤드들을 이어 붙인 context_vec (마지막 차원 = d_out)
        self.out_proj = nn.Linear(d_out, d_out)  # 헤드의 출력을 결합할 Linear 층
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",  # [수정] 굽은 따옴표 → 곧은 따옴표, 빠진 닫는 괄호 추가
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 텐서 크기를 각각의 변수에 담기

        # [수정] W_key(x) → self.W_key(x) (클래스 안의 층은 self.로 불러야 함)
        keys = self.W_key(x)        # (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # d_out 차원을 num_heads와 head_dim 두 개의 차원으로 나눔
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)        # (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)        # (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 축이 4개라 마지막 두 축(2, 3)만 바꿈 → 헤드마다 (num_tokens, num_tokens) 점수 행렬
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)

        # [수정] 두 가지
        #   1) masked_fill → masked_fill_ : 밑줄 없는 버전은 새 텐서를 돌려주기만 해서 결과를 버리면 마스킹이 안 됨
        #   2) self.mask.bool([:num_tokens, :num_tokens]) → self.mask.bool()[:num_tokens, :num_tokens]
        #      bool()을 먼저 호출한 뒤 슬라이싱. mask의 1(미래 토큰) 자리를 -inf로 채움
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)

        # [수정] contigious → contiguous (오타), d_out → self.d_out (forward 안에는 d_out 변수가 없음)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # 헤드들을 이어 붙임
        context_vec = self.out_proj(context_vec)
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)

    # 노트 예시와 같은 입력: (2, 6, 3) = (배치, 토큰 수, 벡터 길이)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    batch = torch.stack((inputs, inputs), dim=0)
    print("batch.shape:", tuple(batch.shape))

    # CausalAttention
    ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    out = ca(batch)
    print("CausalAttention 출력:", tuple(out.shape))

    # MultiHeadAttention (d_out=2, num_heads=2 → head_dim=1)
    mha = MultiHeadAttention(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)
    out = mha(batch)
    print("MultiHeadAttention 출력:", tuple(out.shape))

    # 마스킹 확인: 첫 토큰은 자기 자신만 볼 수 있어야 함 → 첫 행 가중치는 [1, 0, 0, 0, 0, 0]
    b, n, _ = batch.shape
    q = mha.W_query(batch).view(b, n, 2, 1).transpose(1, 2)
    k = mha.W_key(batch).view(b, n, 2, 1).transpose(1, 2)
    s = (q @ k.transpose(2, 3)).masked_fill(mha.mask.bool()[:n, :n], -torch.inf)
    w = torch.softmax(s / k.shape[-1] ** 0.5, dim=-1)
    print("헤드 0의 어텐션 가중치 (윗삼각이 0이어야 함):")
    print(w[0, 0].detach().numpy().round(3))
    print("각 행의 합:", w[0, 0].sum(dim=-1).detach().numpy().round(3))
