# LLM-

『밑바닥부터 만들면서 배우는 LLM』(세바스티안 라시카, 원서 *Build a Large Language Model (From Scratch)*)을 공부하며 장마다 직접 구현하고 정리하는 저장소입니다.
토크나이저부터 셀프 어텐션, GPT 구조, 사전학습, 파인튜닝까지 PyTorch로 한 줄씩 만들어 봅니다.

## 진행 현황

| 장 | 주제 | 폴더 | 상태 |
|---|---|---|---|
| 2 | 텍스트 데이터 다루기 (토큰화, BPE, 슬라이딩 윈도우, 임베딩) | [`ch02_tokenizer/`](ch02_tokenizer/) | 완료 (다시 짜기 일부 보완 중) |
| 3 | 어텐션 메커니즘 (셀프 어텐션, 코잘 마스크, 멀티 헤드) | [`ch03_attention/`](ch03_attention/) | 실습 완료 (정리 노트 작성 중) |
| 4 | GPT 모델 구현 | | |
| 5 | 사전학습 | | |
| 6 | 분류 파인튜닝 | | |
| 7 | 지시를 따르는 파인튜닝 | | |

## 공부 방식

1. 책을 보며 노트북에 코드를 **직접 따라 치고**, 출력을 책과 비교
2. 장을 마치면 핵심 클래스를 **보지 않고 다시 구현** (`.py`)
3. 개념을 **내 말로** `notes.md`에 정리
4. 막힌 곳과 찾은 버그를 기록

## 2장 기록

- 2.2~2.8절을 모두 구현하고, 최종 LLM 입력 `(배치 8, 토큰 4, 임베딩 256)`까지 확인
- 찾아서 고친 버그
  - **구두점이 사라짐**: `re.split`의 정규식에 캡처 그룹 `( )`이 빠져, 나누는 기준인 구두점이 토큰에서 버려짐 (토큰 3,788개 → 고친 뒤 책과 같은 4,690개). 에러 없이 결과만 달라지는 버그라 출력을 책과 비교해 발견
  - **실행할 때마다 토큰 번호가 바뀜**: 어휘 사전을 `set`의 순서대로 번호 매김. `sorted(set(...))`로 정렬해 항상 같은 번호(어휘 사전 1,130개)가 나오도록 수정
  - 2.4절 정규식의 오타(`[[`, 빠진 `:;_`)와 2.7절의 불필요한 `torch.tensor` 감싸기 수정
- 2.2~2.4절 출력은 수정한 코드로 다시 실행한 결과

## 3장 기록

- 3.3~3.6절을 모두 구현: 가중치 없는 셀프 어텐션 → 스케일드 점곱 어텐션(`SelfAttention_v1`, `v2`) → 코잘 마스크와 드롭아웃(`CausalAttention`) → 멀티 헤드(`MultiHeadAttentionWrapper`, `MultiHeadAttention`)
- 확인 셀에서 책의 출력값과 소수점 셋째 자리까지 비교해 모두 일치 (예: `context_vec_2` = `[0.3061, 0.8210]`, 코잘 가중치 `[0.5517, 0.4483]`)
- 코잘 테스트: 뒤쪽 토큰을 바꿔도 앞쪽 토큰의 출력이 그대로인지 확인해, 미래 단어를 보지 않음을 검증
- 찾아서 고친 버그
  - **마스크가 잘못된 칸을 가림**: `torch.triu(..., diagonal=-1)`로 써서 자기 자신과 바로 앞 단어까지 가려짐. 첫 행이 전부 `-inf`가 되어 softmax 결과가 `nan`이 되는 문제라 `diagonal=1`로 수정 (`diagnal` 오타도 함께 수정)
  - **헤드 차원이 맞지 않음**: `MultiHeadAttention`에서 `keys`만 `transpose(1, 2)`하고 `queries`, `values`는 하지 않아 `(b, num_tokens, num_heads, head_dim)`과 `(b, num_heads, ...)`를 곱하려다 에러. 셋 다 `(b, num_heads, num_tokens, head_dim)`으로 바꾸도록 수정. 토큰 수와 헤드 수가 우연히 같으면 에러 없이 틀린 값이 나오는 버그라, 줄마다 텐서 모양을 주석으로 적으며 확인

## 폴더 구성

```
ch02_tokenizer/
├── ch02_practice.ipynb   # 따라 치기 노트북 (확인 셀 포함)
├── notes.md              # 내 말로 정리한 핵심 개념
└── the-verdict.txt       # 2장 실습 데이터

ch03_attention/
├── ch03_practice.ipynb   # 따라 치기 노트북 (책의 출력값과 숫자까지 비교하는 확인 셀)
└── notes.md              # 내 말로 정리할 틀 (텐서 모양 따라가기 표 포함)
```

## 실행 환경

Google Colab 또는 로컬 Python 3.10 이상

```bash
pip install -r requirements.txt
```

## 관련 저장소

- [nsmc-sentiment-classification](https://github.com/uisang2004-maker/nsmc-sentiment-classification): KcBERT 파인튜닝 감성 분류
- [klue-ner-bert](https://github.com/uisang2004-maker/klue-ner-bert): klue/bert-base 개체명 인식

## 참고와 라이선스

- 책의 공식 코드: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache License 2.0). 이 저장소의 코드는 책을 따라 공부하며 직접 작성한 것입니다.
- `the-verdict.txt`: 이디스 워튼의 단편 소설 *The Verdict* (1908), 퍼블릭 도메인. 책의 공식 저장소에서 가져왔습니다.
