# 밸류에이션 팩터 → 기대수익률 모델 (학습 파이프라인)

기술명세서(technical-spec.md)에서 정한 설계를 코드로 구현한 것입니다: 밸류·퀄리티·기술 지표로 섹터 상대 percentile 스코어를 만들고, 그 스코어가 실제로 forward return과 관계가 있는지 통계적으로 검증한 뒤, Ridge(주 모델)/Fama-MacBeth/XGBoost/RandomForest로 기대수익률을 추정합니다.

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 파이프라인 로직 자체를 검증하는 synthetic 데이터 스모크 테스트 (지금 바로 실행 가능, 외부 API 불필요)
python tests/synthetic_smoke_test.py

# 2) 실제 데이터로 돌리려면 (Finnhub API 키 필요: https://finnhub.io/register)
export FINNHUB_API_KEY=your_key_here
python train.py --tickers AAPL MSFT JNJ ...   # 아직 CLI 미구현, 아래 "다음 단계" 참고
```

**Colab에서 실행할 때:** `src/`, `tests/` 폴더 구분 없이 `.py` 파일 전부를 `/content/` 한 폴더에 평평하게 업로드한 뒤 `!python synthetic_smoke_test.py`로 실행하면 됩니다 (내부 import가 `from config import ...` 식이라 같은 폴더에 있으면 그대로 동작). 세션이 끊기면 업로드한 파일이 사라지니 새 세션마다 다시 올려야 합니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `src/config.py` | 지표 목록, 예측 호라이즌, 방향(higher/lower_is_better) 정의 |
| `src/data_collection.py` | yfinance(가격) + Finnhub(과거 point-in-time 펀더멘털) 수집 |
| `src/features.py` | look-ahead bias 없이 시점별 지표값·섹터 percentile·forward return 계산 |
| `src/factor_validation.py` | 분위별 팩터 스프레드, 정보계수(IC), FDR 보정 |
| `src/models.py` | RidgeQuantileModel(주 모델), FamaMacBethModel, XGBQuantileModel, RandomForestQuantileModel |
| `src/pipeline.py` | 위 전부를 시간 기준 Train/Val/Test 분할로 묶어서 실행 |
| `tests/synthetic_smoke_test.py` | 가짜 데이터로 전체 파이프라인이 에러 없이 도는지 + 통계가 말이 되는지 확인 |

## 지금 실제로 검증된 것 / 아직 안 된 것

**검증됨 (synthetic 데이터로 실행 확인)**
- look-ahead bias 없는 point-in-time 피처 계산 (재무제표는 발표 후 45일 지연 반영 — `features.py`의 `REPORTING_LAG_DAYS`, 실제 공시 지연과 다를 수 있어 검증 필요)
- 섹터 상대 percentile + composite score 계산
- 시간 기준 Train/Val/Test 분할 (랜덤 split 아님)
- Ridge/Fama-MacBeth/XGBoost/RandomForest 4개 모델이 전부 에러 없이 학습·예측
- 분위별 팩터 스프레드, IC, FDR 보정까지 전체 파이프라인 실행

**synthetic 테스트에서 실제로 발견된 것 (중요, 실제 데이터에서도 그대로 나올 가능성 높음)**
- **모델 간 과적합 정도 차이가 설계 의도대로 나타남**: XGBoost는 Train MAE가 Val MAE보다 훨씬 낮게 나오는 반면(과적합), Ridge/RandomForest는 Train-Val 격차가 거의 없거나 오히려 Val이 더 낮게 나옴. 3.5절에서 예상한 대로 "작은 데이터에서는 Ridge/RF가 XGBoost보다 안전하다"는 게 실제로 재현됨.
- **다중공선성 때문에 Fama-MacBeth 개별 feature 계수의 부호가 불안정함**: synthetic 데이터는 지표들이 전부 하나의 공통 요인에서 파생되도록 설계했는데, 이렇게 지표끼리 상관관계가 높으면 여러 feature를 한꺼번에 넣은 회귀에서 개별 계수 부호가 뒤집히거나 유의하지 않게 나옵니다(공동 설명력은 있어도 "이 지표가 범인이다"를 개별적으로 짚어내기 어려움). composite_score의 IC 자체는 기대대로 전부 양수였습니다. **실제 지표(PER-PBR, ROE-영업이익률 등)도 서로 상관관계가 높을 가능성이 커서, 실제 데이터에서도 같은 문제가 나올 수 있습니다.** 이 경우 지표를 하나씩 넣는 단변량(univariate) Fama-MacBeth로 바꾸거나, PCA로 차원을 줄이는 방법을 고려해야 합니다 — 지금 코드에는 아직 안 넣었습니다.
- 종목 30개 × 분기 13개(Train 기준)짜리 synthetic 데이터에서도 통계적 유의성(p<0.05)은 안 나왔습니다. 실제 프로젝트 목표인 10~20개 종목이면 검정력이 이보다도 낮을 가능성이 높다는 뜻이므로, "통계적으로 유의미하다"고 말하려면 종목 수/기간을 늘리는 게 중요해집니다.

**실제 Finnhub 계정으로 검증 완료 (2026-09-14, AAPL 기준)**
- `FINNHUB_FIELD_MAP`의 5개 지표(`trailing_pe`, `price_to_book`, `return_on_equity`, `debt_to_equity`, `operating_margin`)는 실제 응답 필드명으로 수정 완료 (`peTTM`, `pb`, `roeTTM`, `totalDebtToEquity`, `operatingMargin`)
- `dividend_yield`, `revenue_growth_yoy`는 Finnhub에 직접 필드가 없다는 것도 확인됨 → 원재료(`eps`, `payoutRatioTTM`, `salesPerShare`)를 대신 수집해서 `features.py`에서 파생 계산하도록 구현:
  - `dividend_yield` ≈ `payout_ratio_ttm * eps / 그 시점 가격` (`features.py`의 `_dividend_yield`)
  - `revenue_growth_yoy` ≈ `sales_per_share`의 YoY(365일±45일 허용) 변화율 (`features.py`의 `add_revenue_growth_yoy`)
  - synthetic 데이터로 파생 로직 자체도 검증함 (`tests/synthetic_smoke_test.py`): 설계한 성장률과 복원된 값이 근접, `dividend_yield`/`revenue_growth_yoy` NA율 0%, 전체 파이프라인 정상 실행 + IC 양수(PASS)

**아직 안 됨 (실제 데이터 연결 필요)**
- 실제 종목 리스트(10~20개) 선정, 실제 데이터 수집 실행, 실제 결과 해석은 안 함 — synthetic 데이터는 로직 검증용일 뿐 실제 시장에 대해 아무것도 말해주지 않음
- CLI(`train.py`)나 MLflow 연동은 아직 없음 — 지금은 `pipeline.run_all_horizons(panel, horizons)`을 직접 호출하는 라이브러리 형태
- `dividend_yield`/`revenue_growth_yoy` 파생 로직은 synthetic 데이터로만 검증됨 — 실제 AAPL 등 데이터로 계산했을 때 값 자체가 상식적인 범위(예: 배당수익률이 0~10% 사이 등)인지는 아직 확인 안 함

## 다음 단계

1. 지원 종목 10~20개 확정 (기술명세서 11절) → `data_collection.py`로 실제 수집
2. 실제 데이터로 `dividend_yield`/`revenue_growth_yoy` 파생값이 상식적인 범위인지 확인
3. 다중공선성 문제 확인되면 univariate Fama-MacBeth 또는 지표 축소 적용
4. MLflow로 실험 추적 연결, `train.py` CLI 작성
