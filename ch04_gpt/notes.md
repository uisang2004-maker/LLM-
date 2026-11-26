# 4장 밑바닥부터 GPT 모델 구현하기: 정리

> 책 문장을 옮기지 말고 내 말로 적기. 다 적은 뒤 "이걸 처음 보는 친구에게 설명할 수 있나?"로 점검하기

## 한 줄 요약

- 4장에서 배운 것:

## 핵심 개념

| 개념 | 내 말로 설명 |
|---|---|
| GPT 모델의 전체 구조 (4.1) | |
| 로짓 | |
| 층 정규화 (LayerNorm) | |
| `scale`, `shift` | |
| GELU (ReLU와 비교) | |
| 피드 포워드 네트워크 | |
| 숏컷(잔차) 연결 | |
| 트랜스포머 블록 | |
| 가중치 묶기 | |
| 텍스트 생성 (탐욕적 디코딩) | |

## 텐서 모양 따라가기

`GPTModel`에서 `batch`(2, 4)가 줄마다 어떤 모양이 되는지 적어 보기 (`GPT_CONFIG_124M`: `emb_dim=768`, `vocab_size=50257`)

| 단계 | 모양 |
|---|---|
| 입력 `in_idx` | (2, 4) |
| `tok_embeds = self.tok_emb(in_idx)` | |
| `pos_embeds = self.pos_emb(torch.arange(seq_len))` | |
| `x = tok_embeds + pos_embeds` | |
| `self.trf_blocks(x)` 뒤 | |
| `self.final_norm(x)` 뒤 | |
| `logits = self.out_head(x)` | (2, 4, 50257) |
| 생성: `logits[:, -1, :]` | |
| 생성: `torch.argmax(..., keepdim=True)` | |
| 생성: `torch.cat((idx, idx_next), dim=1)` | |

## Do it! 공부 노트와 비교

| | Do it! 노트에서 정리한 것 | 4장에서 직접 만들어 보니 |
|---|---|---|
| Add & Norm | 잔차 연결을 더한 뒤 층 정규화 | |
| 피드 포워드 | 위치마다 같은 2층 신경망 | |
| 위치 정보 | 사인·코사인 위치 인코딩 | |
| 디코더 블록 | 마스크드 셀프 어텐션 + 피드 포워드 | |
| GPT 구조 | 트랜스포머의 디코더만 쌓음 | |

## 다시 짜기 체크

- [ ] `LayerNorm`을 보지 않고 다시 짜기
- [ ] `GELU`, `FeedForward`를 보지 않고 다시 짜기
- [ ] `TransformerBlock`을 보지 않고 다시 짜기 (3장의 `MultiHeadAttention`을 `import`해서)
- [ ] `GPTModel`, `generate_text_simple`을 보지 않고 다시 짜기 (`gpt.py`로 저장)
- 막혔던 곳:

## 이해 확인 질문

1. 층 정규화를 하는 이유는? 정규화한 뒤 `scale`과 `shift`를 왜 다시 곱하고 더할까?
2. 층 정규화에서 분산을 구할 때 `unbiased=False`를 쓰는 이유는?
3. 숏컷 연결이 없으면 앞쪽 층의 그레디언트가 왜 작아질까? 숏컷이 있으면 왜 살아날까?
4. 트랜스포머 블록의 입력과 출력 모양이 같아야 하는 이유는?
5. 파라미터가 1억 6,300만 개인데 왜 "GPT-2 124M"이라고 부를까?
6. `generate_text_simple`에서 softmax를 빼고 로짓에 바로 `argmax`를 해도 결과가 같은 이유는? 그리고 생성된 문장이 이상한 이유는?

## 궁금한 점·더 알아볼 것

-
