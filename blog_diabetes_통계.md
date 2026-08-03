# scikit-learn 당뇨병 데이터로 판다스 통계 함수 훑기

> 442명의 환자 데이터로 `describe()`부터 상관계수 3종까지 정리했다.
> 마지막에 예제 코드 하나가 사람을 속이고 있었다는 것도 알게 됐다.

---

## 왜 diabetes 데이터인가

판다스 통계 함수를 연습할 때 데이터 고르는 게 은근히 번거롭다.
파일을 받고, 인코딩 맞추고, 결측치 치우다 보면 정작 통계는 시작도 못 한다.

`load_diabetes()`는 그 과정을 전부 건너뛴다.

- scikit-learn에 **내장**되어 있어 다운로드가 필요 없다
- 결측치가 **하나도 없다**
- 442명 × 10개 특성 + 타깃 1개 — 너무 크지도 작지도 않다
- 이미 **표준화**되어 있어 스케일 고민이 없다

| 컬럼 | 의미 |
|---|---|
| `age` | 나이 (표준화됨) |
| `sex` | 성별 (표준화됨) |
| `bmi` | 체질량 지수 |
| `bp` | 평균 혈압 |
| `s1`~`s6` | 혈청 검사 수치 6종 |
| `target` | 1년 후 당뇨병 진행 정도 |

`s1`~`s6`은 그냥 번호처럼 보이지만 sklearn 공식 설명(`diabetes.DESCR`)에 뜻이 적혀 있다.

| | 의미 |
|---|---|
| `s1` | 총 콜레스테롤 (tc) |
| `s2` | LDL 콜레스테롤 |
| `s3` | **HDL 콜레스테롤** |
| `s4` | 총콜레스테롤 / HDL 비율 |
| `s5` | 혈청 중성지방 (로그 추정) |
| `s6` | 혈당 |

나중에 결과를 해석할 때 이 표가 결정적이었다.

`target`은 0~346 범위의 연속값이다. 분류가 아니라 **회귀** 문제용 데이터다.

---

## 0. 준비 — 한글 폰트부터

그래프에 한글을 쓰면 기본 설정에서는 네모(□□□)로 깨진다.
OS마다 폰트 이름이 달라서 이렇게 분기해 두면 어디서든 돌아간다.

```python
import platform
import matplotlib.pyplot as plt

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')      # 윈도우
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')        # 맥
else:
    plt.rc('font', family='NanumBarunGothic')   # 리눅스

plt.rc('axes', unicode_minus=False)   # 음수 기호 깨짐 방지
```

`unicode_minus=False`를 빼먹으면 안 된다.
한글 폰트로 바꾸는 순간 음수 기호(−)가 따로 깨진다. **항상 짝으로** 넣자.

출력 옵션도 미리 잡아둔다. 열이 11개라 그냥 두면 줄바꿈되어 읽기 힘들다.

```python
pd.set_option('display.expand_frame_repr', False)  # 한 줄로 출력
pd.set_option('display.max_columns', None)         # 열 생략 없이
```

---

## 1. Bunch를 DataFrame으로 — 여기서 한 번 넘어졌다

`load_diabetes()`가 돌려주는 건 DataFrame이 아니다. **Bunch**라는 딕셔너리 비슷한 객체다.

```python
diabetes = load_diabetes()

diabetes.data            # (442, 10) 숫자 배열 — 이름표 없음
diabetes.feature_names   # ['age', 'sex', 'bmi', ...] 열 이름 리스트
diabetes.target          # (442,) 정답값
```

데이터와 열 이름이 **따로 놀고 있다.** 이 둘을 짝지어야 표가 된다.

```python
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
#                 └─ 값            └─ 열 이름
df['target'] = diabetes.target
```

`columns=`를 빼면 열 이름이 `0, 1, 2, ...` 숫자로 자동 부여된다.
`df['bmi']` 대신 `df[2]`를 써야 하고, 몇 번째가 bmi인지 외워야 한다.

### 겪은 오류: `AttributeError: 'DataFrame' object has no attribute 'data'`

위 코드를 실행한 뒤 이렇게 썼다가 막혔다.

```python
print(df.data.shape)     # ❌ AttributeError
print(df.target.shape)   # ✅ 얘는 왜 되지?
```

**원인은 `df`가 이미 DataFrame이 됐다는 것.**
`.data`, `.feature_names`는 Bunch에만 있는 속성이다. DataFrame에는 없다.

```python
diabetes.data            # ✅ Bunch 의 속성
df.data                  # ❌ 없음
df.shape                 # ✅ DataFrame 은 이걸 쓴다
```

재미있는 건 **`df.target`은 오류가 나지 않는다**는 점이다.
판다스는 속성 접근을 만나면 **먼저 열 이름인지 확인**한다.
`target`이라는 열을 방금 만들었으니 `df.target`은 `df['target']`으로 해석된다.
`data`라는 열은 없으니 거기서만 멈춘 것이다.

에러 메시지의 마지막 두 줄이 그 과정을 그대로 보여준다.

```
6320             return self[name]                          ← 열 이름이면 그 열 반환
6321         return object.__getattribute__(self, name)      ← 아니면 속성 조회 → 실패
```

> **교훈:** 속성 접근(`df.target`)은 열 이름에 공백·한글이 있거나
> `count`, `shape`처럼 기존 메서드명과 겹치면 조용히 깨진다.
> `df['target']` 대괄호 방식이 항상 안전하다.

---

## 2. EDA — 데이터 얼굴 익히기

```python
df.head()        # 위에서 5개
df.tail(3)       # 아래에서 3개
df.sample(3)     # 무작위 3개
df.info()        # 타입·결측치
df.describe()    # 요약 통계
```

`sample()`의 기본값은 **1개**다. `head()`/`tail()`이 5개인 것과 다르니 주의.

`info()`로 확인해 보면 결측치가 없고 전부 `float64`다.
전처리 없이 바로 통계로 넘어갈 수 있다는 뜻이다.

`target`의 분포는 이렇다.

```
count    442.0
mean     152.1
std       77.1
min       25.0
50%      140.5
max      346.0
```

평균 152, 표준편차 77. 편차가 꽤 크다.
중앙값(140.5)이 평균(152.1)보다 작으니 **오른쪽으로 살짝 치우친** 분포다.

---

## 3. 인덱싱 5종 — 헷갈리는 것들 정리

| 인덱서 | 지정 방식 | 슬라이싱 끝값 |
|---|---|---|
| `df[]` | 열 이름 또는 행 슬라이싱 | **미포함** |
| `df.loc[행, 열]` | 라벨(이름) 기준 | **포함** |
| `df.iloc[행, 열]` | 위치(번호) 기준 | **미포함** |
| `df.at[행, 열]` | 단일 값, 라벨 | 슬라이싱 불가 |
| `df.iat[행, 열]` | 단일 값, 번호 | 슬라이싱 불가 |

가장 자주 실수하는 건 `loc`의 슬라이싱이다.

```python
df.loc[:5]    # 0~5 → 6개 (끝 포함!)
df.iloc[:5]   # 0~4 → 5개 (끝 미포함)
```

`loc`은 **이름**을 다룬다. "0번부터 5번까지"라고 말할 때 5번을 빼는 건 자연스럽지 않다.
`iloc`은 **위치**를 다루므로 파이썬 리스트 슬라이싱과 같다.

`at`/`iat`은 값 하나만 빠르게 꺼낼 때 쓴다.

```python
df.at[0, 'bmi']   # 0.061696  (이름 기준)
df.iat[0, 2]      # 0.061696  (번호 기준, bmi가 2번째 열)
```

---

## 4. 조건 검색 — `isin()`과 `query()`

### `isin()` — 여러 값 중 포함되는지

SQL의 `IN (...)`과 같다.

```python
df[df['bp'].isin([0.021872, 0.059744, -0.081413])]
df[~df['bp'].isin([...])]   # ~ 를 붙이면 반대
```

### `query()` — 문자열로 조건 쓰기

```python
df.query('bmi > 0.05 and bp > 0')          # & 대신 and
df.query('s1 < 0 or s2 > 0.05')            # | 대신 or
df.query('not(sex == 0.050680)')           # ~ 대신 not()

threshold = 0.05
df.query('bmi > @threshold')               # 변수는 @ 를 붙인다
```

**여기서 규칙이 뒤집힌다는 게 함정이다.**

| | 논리 연산자 |
|---|---|
| `df[...]` 방식 | `&`, `\|`, `~` |
| `query()` 안 | `and`, `or`, `not()` |

`query()`는 문자열을 자체 문법으로 해석하기 때문에 파이썬 키워드를 그대로 쓴다.
정반대라서 처음엔 계속 헷갈렸다.

---

## 5. 통계 함수와 IQR 이상치 탐지

기본 함수들은 이름 그대로다.

```python
df.count()      # 결측 제외 개수
df.mean()       # 평균
df.median()     # 중앙값
df.std()        # 표준편차
df.var()        # 분산
df.quantile([0.25, 0.5, 0.75])
df.corr()       # 상관계수
df.cov()        # 공분산
```

### IQR로 이상치 찾기

사분위 범위(IQR = Q3 − Q1)를 이용한 고전적인 방법이다.
Q1에서 1.5×IQR보다 아래, Q3에서 1.5×IQR보다 위를 이상치로 본다.

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

mask = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
print(mask.sum())          # 열별 이상치 개수
print(mask.sum().sum())    # 전체 개수
```

결과는 이랬다.

```
bmi     3
s1      8
s2      7
s3      7
s4      2
s5      4
s6      9
────────────
총 40개
```

442행 × 11열 = 4,862개 값 중 **40개**니까 0.8% 수준이다.
`age`, `sex`, `bp`, `target`에는 이상치가 없다.
혈청 검사 수치(`s1`~`s6`)에 몰려 있는 게 눈에 띈다.

여기서 `mask.sum()`은 **열별** 합계, `mask.sum().sum()`은 **전체** 합계다.
`sum()`을 두 번 쓰는 이유는 첫 번째가 Series를 돌려주기 때문이다.

---

## 6. 상관계수 3종 — 그리고 예제의 함정

판다스 `corr()`은 세 가지 방법을 지원한다.

| | Pearson | Spearman | Kendall |
|---|---|---|---|
| 측정 대상 | **선형** 관계 | **단조** 관계 | **단조** 관계 |
| 계산 기반 | 값 자체 | 순위 | 순서쌍 일치/불일치 |
| 이상치 영향 | 매우 큼 | 중간 | 가장 적음 |

### 여기서 예제가 나를 속였다

교재 예제는 이렇게 되어 있었다.

```python
data = {
    'x':           [10, 20, 30, 40, 50],
    'y_linear':    [15, 25, 35, 45, 55],   # 선형
    'y_monotonic': [1, 2, 3, 6, 10],       # 단조 증가 (비선형)
    'y_noisy':     [10, 22, 29, 41, 100],  # 이상치 포함
}

pearson_corr, _  = pearsonr(df['x'], df['y_linear'])       # 1.0
spearman_corr, _ = spearmanr(df['x'], df['y_monotonic'])   # 1.0
kendall_corr, _  = kendalltau(df['x'], df['y_noisy'])      # 1.0
```

그리고 설명은 이랬다.

> y_linear는 x와 완전한 선형 관계 → Pearson = 1
> y_monotonic은 비선형이지만 증가하는 관계 → Spearman = 1
> y_noisy는 이상치가 포함되어 있을때 관계 → Kendall = 1

읽으면 **"각 계수가 자기 상황에서만 1이 나온다"**고 이해하게 된다.
그런데 이상했다. 왜 계수마다 다른 열을 쓰지?

**세 계수를 세 열에 전부 적용해 봤다.**

```python
for col in ['y_linear', 'y_monotonic', 'y_noisy']:
    p = pearsonr(df['x'], df[col])[0]
    s = spearmanr(df['x'], df[col])[0]
    k = kendalltau(df['x'], df[col])[0]
    print(f"{col:12s} {p:7.3f} {s:7.3f} {k:7.3f}")
```

```
              Pearson  Spearman  Kendall
y_linear        1.000     1.000    1.000
y_monotonic     0.954     1.000    1.000
y_noisy         0.895     1.000    1.000
```

**Spearman과 Kendall은 세 경우 모두 1.0이었다.**

당연했다. 세 열 모두 x가 커질수록 y도 커지는 **단조 증가**니까,
순위만 보는 두 계수에게는 전부 "완벽한 관계"다.

원래 예제는 계수마다 하나씩만 계산해서, 마치 각 계수가 특정 상황 전용인 것처럼 보이게 했다.
**진짜 배울 점은 세로가 아니라 가로에 있었다.**

- `y_linear` → 셋 다 1.0. 완전한 선형이면 모두 만점
- `y_monotonic` → Pearson만 **0.954로 떨어진다.** 관계가 휘어 있어서
- `y_noisy` → Pearson이 **0.895까지 더 떨어진다.** 이상치 100 하나 때문에

즉 **Pearson만 홀로 무너지고, 순위 기반 두 계수는 끄떡없다**는 게 요점이다.
"이상치가 있으면 Spearman을 쓰라"는 말의 근거가 이 표에 그대로 있다.

---

## 7. 그래서 당뇨병 진행도와 관련 있는 건 뭘까

기왕 상관계수를 배웠으니 실제로 써봤다.

```python
df.corr()['target'].drop('target').sort_values(ascending=False)
```

```
bmi     0.586
s5      0.566
bp      0.441
s4      0.430
s6      0.382
s1      0.212
age     0.188
s2      0.174
sex     0.043
s3     -0.395
```

**BMI(체질량 지수)가 0.586으로 1위다.** 혈청 수치 `s5`가 0.566으로 바짝 뒤따른다.
혈압(`bp`)도 0.441로 상당하다.

흥미로운 건 두 가지다.

**`sex`는 0.043으로 사실상 무관하다.** 성별로는 당뇨병 진행도를 설명할 수 없다는 뜻이다.

**`s3`만 −0.395로 음의 상관이다.** 이 수치가 높을수록 진행도가 낮아진다.
공식 설명을 보면 `s3`는 **HDL 콜레스테롤**이다. 이른바 '좋은 콜레스테롤'이니
방향이 반대인 게 납득이 간다. 열 이름이 `s3`라서 그냥 지나칠 뻔했는데,
`diabetes.DESCR`을 읽어보고서야 결과가 해석됐다.

특성끼리의 상관도 봤다.

```
s1 - s2    0.897
s3 - s4    0.738
s2 - s4    0.660
s4 - s5    0.618
```

`s1`과 `s2`는 0.897로 거의 같이 움직인다.
혈청 검사 항목끼리 강하게 얽혀 있다는 뜻인데,
회귀 모델을 만든다면 **다중공선성**을 신경 써야 할 지점이다.

> 물론 상관관계는 인과관계가 아니다.
> "BMI가 높아서 당뇨병이 진행된다"가 아니라
> "BMI와 진행도가 함께 움직인다"까지만 말할 수 있다.

---

## 마무리

정리하면서 남은 것들.

**1. 객체의 정체를 항상 확인하자.**
`Bunch`와 `DataFrame`은 담고 있는 값이 같아도 **속성 이름이 다르다.**
`AttributeError`가 나면 "이게 지금 무슨 타입이지?"부터 물어야 한다.
`type(df)` 한 줄이면 끝날 문제였다.

**2. `df.target`이 되는 건 우연이다.**
열 이름과 속성 접근이 겹쳐서 통했을 뿐이다. 대괄호를 쓰자.

**3. 예제 코드는 의심하면서 보자.**
상관계수 예제는 실행하면 설명대로 나온다. 오류도 없다.
하지만 **계수마다 다른 데이터를 쓴 것**이 결론을 오해하게 만들었다.
직접 조합을 바꿔 돌려보지 않았다면 그냥 외우고 넘어갔을 것이다.

**4. 통계 함수는 결국 조합이다.**
IQR 이상치 탐지도 `quantile()`, 비교 연산, `|`, `sum()`을 엮은 것뿐이다.
함수 하나하나를 외우기보다 **무엇을 반환하는지**를 아는 게 중요했다.
`mask.sum().sum()`에서 `sum()`이 두 번인 이유처럼.

---

### 다음에 해볼 것

- `target`을 구간으로 나눠(`pd.cut`) 그룹별 특성 비교
- `s1`~`s6` 간 다중공선성 확인 (VIF)
- BMI 하나만으로 단순 선형회귀를 돌려보고 R² 확인

*환경: Python 3.10 · pandas 2.3.3 · scikit-learn 1.7.2 · scipy 1.15.3*
