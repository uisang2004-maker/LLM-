# LLM-

『밑바닥부터 만들면서 배우는 LLM』(세바스티안 라시카, 원서 *Build a Large Language Model (From Scratch)*)을 공부하며 장마다 직접 구현하고 정리하는 저장소입니다.
토크나이저부터 셀프 어텐션, GPT 구조, 사전학습, 파인튜닝까지 PyTorch로 한 줄씩 만들어 봅니다.

## 진행 현황

| 장 | 주제 | 폴더 | 상태 |
|---|---|---|---|
| 2 | 텍스트 데이터 다루기 (토큰화, BPE, 슬라이딩 윈도우, 임베딩) | [`ch02_tokenizer/`](ch02_tokenizer/) | 완료 (다시 짜기 일부 보완 중) |
| 3 | 어텐션 메커니즘 (셀프 어텐션, 코잘 마스크, 멀티 헤드) | [`ch03_attention/`](ch03_attention/) | 실습·다시 짜기 완료 (정리 노트 보완 중) |
| 4 | GPT 모델 구현 (층 정규화, GELU, 숏컷 연결, 트랜스포머 블록, 텍스트 생성) | [`ch04_gpt/`](ch04_gpt/) | 진행 중 |
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
- 보지 않고 다시 짜기: `CausalAttention`, `MultiHeadAttention`을 노트에 다시 짜고 `attention.py`로 옮겨 실행. 입력 `(2, 6, 3)` → 출력 `(2, 6, 2)`가 노트의 "텐서 모양 따라가기" 표와 일치하고, 코잘 가중치의 윗삼각이 모두 0·각 행의 합이 1인 것까지 확인
- 다시 짜기에서 찾아서 고친 버그 (고친 줄마다 `attention.py`에 `# [수정]` 주석)
  - **마스킹이 적용되지 않음**: `attn_scores.masked_fill(...)`처럼 밑줄 없는 버전을 써서, 마스킹된 새 텐서를 만들기만 하고 버림. 에러 없이 미래 단어를 보게 되는 버그라 `masked_fill_`(제자리 연산)로 수정
  - **3차원 텐서에 `.T`**: `CausalAttention`에서 `queries @ key.T`로 써서 배치 축까지 뒤집힘(`(d_out, num_tokens, b)`). 마지막 두 축만 바꾸는 `keys.transpose(1, 2)`로 수정 (`.T`는 역행렬이 아니라 전치이고, 멀티 헤드에서는 축이 4개라 `transpose(2, 3)`)
  - **출력 투영 층의 입력 차원**: `out_proj = nn.Linear(d_in, d_out)`으로 써서, 헤드를 이어 붙인 컨텍스트 벡터(마지막 차원 `d_out`)와 맞지 않음. `nn.Linear(d_out, d_out)`으로 수정 (`d_in`과 `d_out`이 우연히 같으면 에러가 나지 않음)
  - 그 밖에 `self.num_heads` 저장 누락, `self.` 없이 층 호출, `W_values`/`W_value` 이름 불일치, `contigious` 오타, 굽은 따옴표(`’mask’`) 수정
- 정리 노트 검토: `notes_review.md`에 Claude가 단 첨삭(맞는 것·보완할 것·틀린 것)을 원래 노트와 나란히 정리

## 폴더 구성

```
ch02_tokenizer/
├── ch02_practice.ipynb   # 따라 치기 노트북 (확인 셀 포함)
├── notes.md              # 내 말로 정리한 핵심 개념
└── the-verdict.txt       # 2장 실습 데이터

ch03_attention/
├── ch03_practice.ipynb   # 따라 치기 노트북 (책의 출력값과 숫자까지 비교하는 확인 셀)
├── attention.py          # 보지 않고 다시 짠 CausalAttention, MultiHeadAttention (python attention.py로 모양·마스킹 확인)
├── notes.md              # 내 말로 정리한 핵심 개념, 텐서 모양 따라가기, 이해 확인 질문
└── notes_review.md       # notes.md 첨삭 (Claude 검토)

ch04_gpt/
├── ch04_practice.ipynb   # 따라 치기 노트북 (3장의 attention.py를 불러와 GPT 조립, 파라미터 수·생성 토큰까지 책과 비교)
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
