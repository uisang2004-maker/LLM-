# 2장 텍스트 데이터 다루기: 정리

> 책 문장을 옮기지 말고 내 말로 적기. 다 적은 뒤 "이걸 처음 보는 친구에게 설명할 수 있나?"로 점검하기

## 한 줄 요약

- 2장에서 배운 것:

## 핵심 개념

| 개념 | 내 말로 설명 |
|---|---|
| 임베딩 | 날것의 텍스트를 연속적인 수치 벡터 표현으로 바꾸는 것 |
| 토큰화 | 텍스트를 서브워드/단어 등의 단위로 쪼개어 나타내는 것 |
| 토큰 ID와 어휘 사전 | 토큰화된 단어들을 어휘 사전에 각각 저장함. 이후 입력된 단어들을 어휘 사전에 등록된 인덱스값(토큰 ID)으로 바꿈 |
| `<\|unk\|>`, `<\|endoftext\|>` | 모델이 알지 못하는 단어나 텍스트를 처리할 수 있는 미등록 토큰과, 관련 없는 문장들을 구분해 주는 스페셜 토큰임 |
| BPE | 바이트 페어 인코딩은 텍스트를 서브워드 단위로 토큰화함. 알지 못하는 단어를 스페셜 토큰 없이도 이해하고 처리 가능 |
| 슬라이딩 윈도우 (입력과 정답) | 슬라이딩 윈도우 기법은 텍스트 위를 윈도우가 일정 스텝(스트라이드)만큼 이동하며 배치를 만듦 |
| `max_length`와 `stride` | max_length는 윈도우가 한 번에 담는 토큰 개수(샘플 하나의 토큰 수)이고 스트라이드는 윈도우가 한 번에 이동하는 크기임. 둘이 같다면 중첩되는 토큰 없이 텍스트를 처리함 |
| 토큰 임베딩 + 위치 임베딩 | 절대 임베딩의 최종 결과는 위치 정보(위치 임베딩)를 포함한 텐서가 된다. 이는 셀프 어텐션이 위치나 순서 정보를 반영하지 못하기 때문이다 |

## Do it! (BERT)와 비교

| | Do it! (BERT) | 밑바닥 LLM (GPT) |
|---|---|---|
| 토크나이저 | 워드피스, 단어 중간 조각에 `##` | BPE (`tiktoken`) |
| 특수 토큰 | `[CLS]`, `[SEP]`, `[PAD]` | `<\|endoftext\|>` |
| 학습 데이터 | 빈칸 맞히기 | 다음 토큰 맞히기 (슬라이딩 윈도우) |
| 모르는 단어 | `[UNK]` | 사용 x |
| 위치 정보 | pos_embedding | pos_embedding |

## 다시 짜기 체크

- [x] `SimpleTokenizerV2`를 보지 않고 다시 짜기

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'[,.?!"()\']|--|\s', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'), r'\1', text)
        return text
```

- [x] `GPTDatasetV1`, `create_dataloader_v1`을 보지 않고 다시 짜기

```python
class GPTDatabaseV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(slef.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
```

> 다시 짠 코드 리뷰 (다음에 다시 짤 때 확인할 점)
> - `SimpleTokenizerV2.encode`에 V2의 핵심인 **사전에 없는 단어를 `<|unk|>`로 바꾸는 줄**이 빠짐
> - `re.split` 정규식에 캡처 그룹 `( )`이 없어서 구두점이 사라짐 (노트북에서 고친 것과 같은 버그). `:`, `;`, `_`도 빠짐
> - `decode`의 정규식에 `]`와 닫는 따옴표가 빠짐: `r'\s+([,.?!"()\'])'`
> - 클래스 이름 `GPTDatabaseV1` → `GPTDatasetV1`, `slef` → `self` 오타
> - `create_dataloader_v1`은 아직 다시 짜지 않음

- 막혔던 곳: `re.split`이 나누는 기준 `? ! , . " ' ( ) -- \s`

## 이해 확인 질문

1. `SimpleTokenizerV1`이 처음 보는 단어에서 에러가 나는 이유는? V2는 어떻게 해결했나?
   - 처음 보는 단어를 처리할 스페셜 토큰의 부재. V2는 스페셜 토큰 `<|unk|>`로 해결함
2. BPE는 왜 `<|unk|>` 토큰이 필요 없을까?
   - 단어를 서브워드 단위로 나누어 처리하기 때문에, 처음 보는 단어라도 개별 서브워드로 나누어서 처리 가능하기 때문이다
3. `stride`를 `max_length`보다 작게 하면 무엇이 달라질까? 장점과 단점은?
   - 배치에 중첩되는 토큰이 더 많이 들어간다. 단어 사이의 미묘한 관계를 더 잘 이해할 수 있지만 계산 효율이 떨어진다+ 과적합의 위험이 커진다
4. 위치 임베딩을 더하지 않으면 모델은 어떤 정보를 잃을까?
   - 단어의 순서와 위치 정보를 읽을 수 없게 된다
5. 슬라이딩 윈도우 데이터셋과 NER의 `NERDataset`은 무엇이 같고 무엇이 다를까?
   - 슬라이딩 윈도우의 데이터셋은 자기 자신의 데이터에서 stride만큼씩 밀어서 target(label)을 생성하는 반면, NERDataset에서는 사람이 직접 붙여 준 label을 사용한다

## 궁금한 점·더 알아볼 것

-
