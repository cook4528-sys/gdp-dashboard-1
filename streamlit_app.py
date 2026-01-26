# streamlit_app.py
# 대기질 관측소 대시보드: [현황] [예측] [알람] 페이지 분리 + 종합 AQI 표시 보강
# 실행: streamlit run streamlit_app.py

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError


# =========================================================
# (옵션) Plotly / Matplotlib (없으면 st.line_chart)
# =========================================================
PLOTLY_OK = True
try:
    import plotly.graph_objects as go  # noqa: F401
except ModuleNotFoundError:
    PLOTLY_OK = False

MPL_OK = True
try:
    import matplotlib.pyplot as plt  # noqa: F401
except ModuleNotFoundError:
    MPL_OK = False

# =========================================================
# (옵션) scikit-learn (없으면 numpy Ridge로 대체)
# =========================================================
SKLEARN_OK = True
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
except ModuleNotFoundError:
    SKLEARN_OK = False


# =========================================================
# Streamlit 기본 설정
# =========================================================
st.set_page_config(page_title="대기질 관측소 대시보드", layout="wide")
st.title("대기질 관측소 대시보드")


# =========================================================
# 전역 상수
# =========================================================
DEFAULT_CANDIDATES = [
    "pollution_2018_2023_3.csv",
    "./data/pollution_2018_2023_3.csv",
    "/mnt/data/pollution_2018_2023_3.csv",
]

POLLUTANTS = ["o3", "no2", "co", "so2"]
AQI_COLS = [f"{p}_aqi" for p in POLLUTANTS]
MEAN_COLS = [f"{p}_mean" for p in POLLUTANTS]
MET_COLS = ["temp_c", "pressure_pa", "met_rain_mm", "met_wind_u", "met_wind_v"]

AQI_BANDS = [
    (0, 50, "좋음(Good)"),
    (51, 100, "보통(Moderate)"),
    (101, 150, "민감군 나쁨(USG)"),
    (151, 200, "나쁨(Unhealthy)"),
    (201, 300, "매우 나쁨(Very Unhealthy)"),
    (301, 500, "위험(Hazardous)"),
]


# =========================================================
# Secrets 안전 접근 (secrets.toml 없어도 앱 실행)
# =========================================================
def get_secret_safe(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)


# =========================================================
# 유틸
# =========================================================
def aqi_category(v: float) -> str:
    if pd.isna(v):
        return "N/A"
    v = float(v)
    for lo, hi, name in AQI_BANDS:
        if lo <= v <= hi:
            return name
    if v < 0:
        return "N/A"
    return "위험(Hazardous)"


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_geometry_point(geo_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    geometry 컬럼: GeoJSON 문자열 가정
      {"type":"Point","coordinates":[lon, lat]}
    """
    if not isinstance(geo_str, str) or not geo_str.strip():
        return None, None
    try:
        obj = json.loads(geo_str)
        coords = obj.get("coordinates", None)
        if isinstance(coords, list) and len(coords) >= 2:
            lon = safe_float(coords[0])
            lat = safe_float(coords[1])
            return lat, lon
    except Exception:
        return None, None
    return None, None


def candidate_default_path() -> Optional[str]:
    for p in DEFAULT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def toast(msg: str, icon: str = "ℹ️"):
    if hasattr(st, "toast"):
        st.toast(msg, icon=icon)


def compute_overall_aqi_row(row: pd.Series) -> float:
    """행 단위 종합 AQI(4개 오염물질 AQI 최대값) - 기존 overall_aqi가 NaN일 때 보강"""
    vals = []
    for c in AQI_COLS:
        v = row.get(c, np.nan)
        if pd.notna(v):
            vals.append(float(v))
    return float(np.nanmax(vals)) if vals else np.nan


def pick_latest_valid_row(df_site: pd.DataFrame, prefer_cols: List[str]) -> pd.Series:
    """
    최신 행이 전체 NaN인 경우가 있어, '종합AQI/오염물질AQI 중 하나라도 유효'한 최신 행을 선택.
    """
    d = df_site.sort_values("date")
    mask = np.zeros(len(d), dtype=bool)
    for c in prefer_cols:
        if c in d.columns:
            mask |= d[c].notna().to_numpy()
    if mask.any():
        return d.loc[mask].iloc[-1]
    return d.iloc[-1]


# =========================================================
# 차트 렌더러(Plotly → Matplotlib → st.line_chart)
# =========================================================
def render_multi_line(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, y_label: str, height: int = 420):
    use_cols = [x_col] + [c for c in y_cols if c in df.columns]
    dfp = df[use_cols].copy().dropna(subset=[x_col])
    if len(dfp) == 0:
        st.info("표시할 데이터가 없습니다.")
        return

    if PLOTLY_OK:
        fig = go.Figure()
        for c in y_cols:
            if c in dfp.columns:
                fig.add_trace(go.Scatter(x=dfp[x_col], y=dfp[c], mode="lines", name=c))
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h"),
            xaxis_title=x_col,
            yaxis_title=y_label,
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    if MPL_OK:
        import matplotlib.pyplot as plt  # local import

        fig, ax = plt.subplots()
        for c in y_cols:
            if c in dfp.columns:
                ax.plot(dfp[x_col], dfp[c], label=c)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_label)
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        return

    st.line_chart(dfp.set_index(x_col)[[c for c in y_cols if c in dfp.columns]], height=height)


def render_single_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, height: int = 240):
    render_multi_line(df, x_col, [y_col], title, y_label, height)


# =========================================================
# 데이터 로드/정규화
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def normalize_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    required = {"site", "city", "county", "state", "date", "geometry"} | set(AQI_COLS) | set(MEAN_COLS)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    for c in AQI_COLS + MEAN_COLS + [c for c in MET_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "is_observed" in df.columns:
        df["is_observed"] = pd.to_numeric(df["is_observed"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_observed"] = 1

    if "is_imputed" in df.columns:
        df["is_imputed"] = pd.to_numeric(df["is_imputed"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_imputed"] = 0

    latlon = df["geometry"].apply(parse_geometry_point)
    df["lat"] = latlon.apply(lambda t: t[0])
    df["lon"] = latlon.apply(lambda t: t[1])

    # 종합 AQI(보수적 운영): 4개 AQI의 최대값
    df["overall_aqi"] = df[AQI_COLS].max(axis=1, skipna=True)

    # 종합 AQI가 NaN으로 남는 케이스 보강(행 단위 재계산)
    # (예: 일부 컬럼이 object로 남았다가 numeric 변환 실패한 경우, 또는 특정 행 AQI 모두 NaN인 경우)
    # -> numeric 변환은 했으므로, 여기서는 "all NaN" 행을 그대로 두되 KPI 선택에서 유효값 행을 우선 선택하도록 처리함.

    def _main_pollutant(row) -> str:
        vals = {p: row.get(f"{p}_aqi", np.nan) for p in POLLUTANTS}
        vals = {k: v for k, v in vals.items() if pd.notna(v)}
        if not vals:
            return "N/A"
        return max(vals, key=vals.get).upper()

    df["main_pollutant"] = df.apply(_main_pollutant, axis=1)
    df["overall_cat"] = df["overall_aqi"].apply(aqi_category)

    df = df.sort_values(["site", "date"]).reset_index(drop=True)
    return df


# =========================================================
# 예측(옵션): sklearn 있으면 HGBR, 없으면 Ridge(선형) fallback
# =========================================================
def make_time_features(dts: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dts)
    out = pd.DataFrame(index=d.index)
    out["dow"] = d.dt.dayofweek.astype(int)
    out["month"] = d.dt.month.astype(int)
    out["doy"] = d.dt.dayofyear.astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def make_supervised(ts: pd.Series, dates: pd.Series, lags: int = 14, roll_windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    df = pd.DataFrame({"y": ts.values}, index=pd.to_datetime(dates))
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = df["y"].shift(k)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std(ddof=0)

    tf = make_time_features(df.index.to_series())
    X = pd.concat([df.drop(columns=["y"]), tf], axis=1)
    y = df["y"]
    out = pd.concat([X, y], axis=1).dropna()
    return out


def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def ridge_fit(A: np.ndarray, b: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    I = np.eye(A.shape[1])
    I[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + alpha * I, A.T @ b)


def ridge_predict(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    return A @ w


@dataclass
class ForecastResult:
    pred_df: pd.DataFrame
    mae: Optional[float]


@st.cache_data(show_spinner=False)
def train_and_forecast_site(df_site: pd.DataFrame, target_col: str, horizon: int, lags: int = 14) -> ForecastResult:
    d = df_site[["date", target_col]].dropna().sort_values("date").copy()
    if len(d) < (lags + 60):
        return ForecastResult(pred_df=pd.DataFrame(columns=["date", "pred"]), mae=None)

    sup = make_supervised(d[target_col], d["date"], lags=lags)
    X = sup.drop(columns=["y"])
    y = sup["y"]

    test_n = min(90, max(30, int(len(sup) * 0.15)))
    X_train, y_train = X.iloc[:-test_n], y.iloc[:-test_n]
    X_test, y_test = X.iloc[-test_n:], y.iloc[-test_n:]

    model = None
    mae = None

    if SKLEARN_OK:
        model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=400, random_state=42)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_hat))
        w = None
        cols = X_train.columns.tolist()
    else:
        cols = X_train.columns.tolist()
        Xt = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        yt = y_train.loc[Xt.index].astype(float)

        A = Xt.to_numpy(dtype=float)
        b = yt.to_numpy(dtype=float)
        A = np.c_[np.ones(len(A)), A]
        w = ridge_fit(A, b, alpha=2.0)

        Xv = X_test.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        Av = Xv.to_numpy(dtype=float)
        Av = np.c_[np.ones(len(Av)), Av]
        y_hat = ridge_predict(Av, w)
        mae = mean_absolute_error_np(y_test.to_numpy(dtype=float), y_hat)

    history = d.set_index("date")[target_col].copy()
    last_date = history.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    preds: List[float] = []
    hist_vals = history.copy()

    def build_row(dt: pd.Timestamp) -> pd.DataFrame:
        row = {}
        for k in range(1, lags + 1):
            row[f"lag_{k}"] = float(hist_vals.iloc[-k]) if len(hist_vals) >= k else np.nan
        for wdw in [3, 7, 14]:
            if len(hist_vals) >= wdw:
                row[f"roll_mean_{wdw}"] = float(hist_vals.iloc[-wdw:].mean())
                row[f"roll_std_{wdw}"] = float(hist_vals.iloc[-wdw:].std(ddof=0))
            else:
                row[f"roll_mean_{wdw}"] = np.nan
                row[f"roll_std_{wdw}"] = np.nan

        tf = make_time_features(pd.Series([dt]))
        for c in tf.columns:
            row[c] = float(tf.iloc[0][c])

        x_row = pd.DataFrame([row])
        for c in cols:
            if c not in x_row.columns:
                x_row[c] = np.nan
        return x_row[cols]

    for dt in future_dates:
        x_row = build_row(dt)
        if x_row.isna().any(axis=1).iloc[0]:
            pred = float(hist_vals.iloc[-1])
        else:
            if SKLEARN_OK and model is not None:
                pred = float(model.predict(x_row)[0])
            else:
                Xp = x_row.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                Ap = Xp.to_numpy(dtype=float)
                Ap = np.c_[np.ones(len(Ap)), Ap]
                pred = float(ridge_predict(Ap, w)[0])
        preds.append(pred)
        hist_vals.loc[dt] = pred

    return ForecastResult(pred_df=pd.DataFrame({"date": future_dates, "pred": preds}), mae=mae)


def make_climatology(df_site: pd.DataFrame, target_col: str) -> pd.Series:
    d = df_site[["date", target_col]].dropna().copy()
    d["doy"] = d["date"].dt.dayofyear
    return d.groupby("doy")[target_col].mean()


def sustained_flags(values: pd.Series, threshold: float) -> pd.Series:
    cnt = 0
    out = []
    for v in values:
        if pd.notna(v) and v >= threshold:
            cnt += 1
        else:
            cnt = 0
        out.append(cnt)
    return pd.Series(out, index=values.index)


# =========================================================
# 알람 평가 + Slack Webhook(선택)
# =========================================================
def send_slack_webhook(webhook_url: str, text: str) -> bool:
    if not webhook_url:
        return False
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def evaluate_alert_state(
    df_site: pd.DataFrame,
    target_col: str,
    alert_threshold: float,
    sustain_days: int,
    anom_threshold: float,
    delta_threshold: float,
    earlywarn_days: int,
    forecast_df: Optional[pd.DataFrame] = None,  # columns: date, pred
) -> Dict[str, object]:
    reasons: List[str] = []
    level = "NORMAL"
    is_alert = False

    d = df_site[["date", target_col]].dropna().sort_values("date").copy()
    if len(d) < max(30, sustain_days + 5):
        return {"level": "NORMAL", "reasons": ["데이터 부족"], "is_alert": False}

    # 1) 임계 초과 + 지속
    d_tail = d.tail(max(30, sustain_days + 10)).copy()
    d_tail["sustain"] = sustained_flags(d_tail[target_col], float(alert_threshold)).astype(int)
    if int(d_tail["sustain"].iloc[-1]) >= int(sustain_days):
        is_alert = True
        reasons.append(f"임계값({alert_threshold:.0f}) 초과 {sustain_days}일 지속")

    # 2) anomaly(클라이마톨로지 대비)
    clim = make_climatology(df_site, target_col)
    last = d.iloc[-1]
    doy = int(pd.to_datetime(last["date"]).dayofyear)
    clim_v = float(clim.get(doy, np.nan))
    if pd.notna(clim_v) and pd.notna(last[target_col]):
        anom = float(last[target_col] - clim_v)
        if anom >= float(anom_threshold):
            reasons.append(f"anomaly +{anom:.1f} (기준 +{anom_threshold:.0f})")
            level = "WATCH"

    # 3) 전일 대비 급등(Δ)
    if len(d) >= 2:
        delta = float(d.iloc[-1][target_col] - d.iloc[-2][target_col])
        if delta >= float(delta_threshold):
            reasons.append(f"전일 대비 +{delta:.1f} (기준 +{delta_threshold:.0f})")
            level = "WATCH"

    # 4) 조기경보(예측 기반)
    if forecast_df is not None and not forecast_df.empty and earlywarn_days > 0:
        f = forecast_df.sort_values("date").head(int(earlywarn_days))
        if (f["pred"] >= float(alert_threshold)).any():
            reasons.append(f"조기경보: {earlywarn_days}일 이내 임계 초과 예측")
            if level == "NORMAL":
                level = "WATCH"

    if is_alert:
        level = "ALERT"

    return {"level": level, "reasons": reasons, "is_alert": is_alert}


# =========================================================
# 사이드바: 공통(데이터/필터/페이지)
# =========================================================
with st.sidebar:
    st.header("페이지")
    page = st.radio("이동", ["현황", "예측", "알람"], index=0)

    st.divider()
    st.header("데이터")
    default_path = candidate_default_path()
    uploaded = st.file_uploader("CSV 업로드(옵션)", type=["csv"])
    if uploaded is None:
        st.caption("업로드가 없으면 경로 입력/기본 경로에서 로드합니다.")
        st.text_input("CSV 경로", value=default_path or "", key="csv_path")
    else:
        st.session_state["csv_path"] = ""

    st.divider()
    st.header("모니터링")
    refresh_sec = st.number_input("자동 갱신(초) - 0이면 OFF", min_value=0, max_value=3600, value=0, step=10)
    if refresh_sec and refresh_sec > 0:
        components.html(f"<meta http-equiv='refresh' content='{int(refresh_sec)}'>", height=0)

# 데이터 로드
try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        csv_path = (st.session_state.get("csv_path") or "").strip()
        if not csv_path:
            if default_path is None:
                st.error("CSV를 업로드하거나 경로를 입력해 주세요.")
                st.stop()
            csv_path = default_path
        raw = load_csv(csv_path)

    df = normalize_data(raw)
except Exception as e:
    st.error(f"데이터 로드/정규화 오류: {e}")
    st.stop()

site_counts = df.groupby("site").size().sort_values(ascending=False)

with st.sidebar:
    st.divider()
    st.header("필터")
    q = st.text_input("관측소 검색(부분일치)", value="")
    show_all_sites = st.checkbox("전체 관측소 목록 표시(느릴 수 있음)", value=False)

    if q.strip():
        options = [s for s in site_counts.index.tolist() if q.lower() in str(s).lower()][:300]
    else:
        options = site_counts.index.tolist() if show_all_sites else site_counts.index[:200].tolist()

    if not options:
        st.warning("검색 결과가 없습니다.")
        st.stop()

    selected_site = st.selectbox("관측소(site)", options=options, index=0)

    df_site_all = df[df["site"] == selected_site].copy()
    min_d = df_site_all["date"].min().date()
    max_d = df_site_all["date"].max().date()

    date_range = st.date_input("기간", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    use_imputed = st.checkbox("보정값(is_imputed=1) 포함", value=True)

    # 예측/알람에서 사용
    horizon = st.slider("예측기간(일)", min_value=7, max_value=30, value=14, step=1)
    target = st.selectbox(
        "예측/감시 지표",
        options=[
            ("overall_aqi", "종합 AQI(최대값 기준)"),
            ("o3_aqi", "O3 AQI"),
            ("no2_aqi", "NO2 AQI"),
            ("co_aqi", "CO AQI"),
            ("so2_aqi", "SO2 AQI"),
        ],
        format_func=lambda x: x[1],
    )[0]

# 기간/보정 포함 반영
mask = (df_site_all["date"].dt.date >= start_date) & (df_site_all["date"].dt.date <= end_date)
df_site = df_site_all.loc[mask].copy()
if not use_imputed:
    df_site = df_site[df_site["is_imputed"] == 0].copy()
df_site = df_site.sort_values("date").reset_index(drop=True)

df_model = df_site_all.copy()
if not use_imputed:
    df_model = df_model[df_model["is_imputed"] == 0].copy()
df_model = df_model.sort_values("date").reset_index(drop=True)

if len(df_model) == 0:
    st.warning("필터 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# 최신 유효 행(종합AQI가 NaN으로 보이는 문제 보강)
latest_row = pick_latest_valid_row(df_model, prefer_cols=["overall_aqi"] + AQI_COLS)
latest_overall = latest_row.get("overall_aqi", np.nan)
if pd.isna(latest_overall):
    latest_overall = compute_overall_aqi_row(latest_row)


# =========================================================
# 공통 KPI(상단)
# =========================================================
site_city = df_site_all["city"].mode().iloc[0] if df_site_all["city"].notna().any() else ""
site_county = df_site_all["county"].mode().iloc[0] if df_site_all["county"].notna().any() else ""
site_state = df_site_all["state"].mode().iloc[0] if df_site_all["state"].notna().any() else ""
lat, lon = latest_row.get("lat", None), latest_row.get("lon", None)

kpi = st.columns([2.2, 2.2, 2.0, 2.0, 3.6])
kpi[0].metric("관측소", selected_site)
kpi[1].metric("지역", f"{site_city}, {site_county}, {site_state}")
kpi[2].metric("기준일", str(pd.to_datetime(latest_row["date"]).date()))
kpi[3].metric("종합 AQI", f"{latest_overall:.0f}" if pd.notna(latest_overall) else "N/A")
kpi[4].metric("상태", f"{aqi_category(latest_overall)} / 주오염: {latest_row.get('main_pollutant','N/A')}")


# =========================================================
# 페이지 1) 현황
# =========================================================
if page == "현황":
    st.subheader("현황")

    left, right = st.columns([2.2, 1.0], gap="large")
    with left:
        st.markdown("**최신 지표**")
        show = pd.DataFrame(
            {
                "지표": ["O3 AQI", "NO2 AQI", "CO AQI", "SO2 AQI", "종합 AQI(최대값)"],
                "값": [
                    latest_row.get("o3_aqi", np.nan),
                    latest_row.get("no2_aqi", np.nan),
                    latest_row.get("co_aqi", np.nan),
                    latest_row.get("so2_aqi", np.nan),
                    latest_overall,
                ],
                "분류": [
                    aqi_category(latest_row.get("o3_aqi", np.nan)),
                    aqi_category(latest_row.get("no2_aqi", np.nan)),
                    aqi_category(latest_row.get("co_aqi", np.nan)),
                    aqi_category(latest_row.get("so2_aqi", np.nan)),
                    aqi_category(latest_overall),
                ],
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("**데이터 품질(선택기간)**")
        imputed_ratio = float(df_site["is_imputed"].mean()) if len(df_site) else 0.0
        observed_ratio = float(df_site["is_observed"].mean()) if len(df_site) else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("레코드 수", f"{len(df_site):,}")
        c2.metric("관측 비율(is_observed=1)", f"{observed_ratio*100:.1f}%")
        c3.metric("보정 비율(is_imputed=1)", f"{imputed_ratio*100:.1f}%")

    with right:
        st.markdown("**관측소 위치**")
        if pd.notna(lat) and pd.notna(lon):
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=10)
            st.caption(f"좌표: {lat:.5f}, {lon:.5f}")
        else:
            st.info("geometry 좌표 정보가 없어 지도 표시가 불가합니다.")

    st.divider()
    st.subheader("추세(시계열)")

    with st.expander("표시 옵션", expanded=False):
        last_n = st.slider("최근 N일(시계열 표시)", min_value=30, max_value=365, value=120, step=10)
        show_means = st.checkbox("Mean(평균 농도)도 표시", value=False)

    if len(df_site) == 0:
        st.warning("선택 기간에 데이터가 없습니다.")
        st.stop()

    df_ts = df_site.sort_values("date").tail(last_n).copy()
    render_multi_line(df_ts, "date", ["overall_aqi"] + AQI_COLS, "AQI 시계열", "AQI", height=420)

    if show_means:
        render_multi_line(df_ts, "date", MEAN_COLS, "Mean(농도) 시계열", "Mean", height=360)

    st.subheader("월별 추세(평균)")
    df_m = df_site[["date", "overall_aqi"] + AQI_COLS].copy()
    df_m["month"] = df_m["date"].dt.to_period("M").dt.to_timestamp()
    df_m_agg = df_m.groupby("month")[["overall_aqi"] + AQI_COLS].mean().reset_index()
    render_multi_line(df_m_agg, "month", ["overall_aqi"] + AQI_COLS, "월별 평균 AQI", "AQI", height=360)

    st.divider()
    st.subheader("다운로드")
    out_df = df_site.copy()
    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="선택기간 데이터 다운로드(CSV)",
        data=csv_bytes,
        file_name=f"air_quality_{selected_site[:40].replace(' ', '_')}_range.csv",
        mime="text/csv",
    )
    st.caption("※ 종합 AQI(overall_aqi)는 O3/NO2/CO/SO2 AQI 중 최대값(보수적 운영)입니다.")


# =========================================================
# 페이지 2) 예측
# =========================================================
elif page == "예측":
    st.subheader("예측")

    st.caption("예측 엔진: scikit-learn" if SKLEARN_OK else "예측 엔진: numpy Ridge(선형) (sklearn 미설치 대체)")

    with st.spinner("예측 데이터 생성 중..."):
        fr = train_and_forecast_site(df_model, target, horizon=int(horizon), lags=14)

    if fr.mae is None or fr.pred_df.empty:
        st.warning("학습 데이터가 부족하여 예측을 생성할 수 없습니다. (기간/관측소 변경 또는 데이터 누적 필요)")
        st.stop()

    st.metric("백테스트 MAE", f"{fr.mae:.2f}")
    pred_df = fr.pred_df.copy()

    # anomaly
    clim = make_climatology(df_model, target)
    pred_df["doy"] = pred_df["date"].dt.dayofyear
    pred_df["climatology"] = pred_df["doy"].map(clim).astype(float)
    pred_df["anomaly"] = pred_df["pred"] - pred_df["climatology"]

    hist = df_model[["date", target]].dropna().sort_values("date").tail(120).rename(columns={target: "actual"})
    merged = hist.merge(pred_df[["date", "pred", "climatology", "anomaly"]], on="date", how="outer").sort_values("date")

    render_multi_line(
        merged,
        "date",
        [c for c in ["actual", "pred", "climatology"] if c in merged.columns],
        "실측 vs 예측 vs 클라이마톨로지",
        target,
        height=420,
    )
    render_single_line(pred_df, "date", "anomaly", "예측 anomaly(예측-평년)", "anomaly", height=240)

    st.divider()
    st.subheader("예측 데이터")
    st.dataframe(pred_df[["date", "pred", "climatology", "anomaly"]], use_container_width=True, hide_index=True)

    csv_bytes = pred_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="예측 데이터 다운로드(CSV)",
        data=csv_bytes,
        file_name=f"forecast_{selected_site[:40].replace(' ', '_')}_{target}.csv",
        mime="text/csv",
    )


# =========================================================
# 페이지 3) 알람
# =========================================================
else:
    st.subheader("알람")

    with st.sidebar:
        st.divider()
        st.header("알람 기준(이상징후)")
        alert_threshold = st.number_input("임계값(AQI)", min_value=0.0, max_value=500.0, value=101.0, step=1.0)
        sustain_days = st.number_input("지속일수(연속)", min_value=1, max_value=14, value=2, step=1)
        anom_threshold = st.number_input("anomaly 임계(+)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
        delta_threshold = st.number_input("전일 대비 급등(Δ) 임계(+)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
        earlywarn_days = st.number_input("조기경보(예측) 윈도우(일)", min_value=0, max_value=30, value=7, step=1)

        st.header("알림 채널(선택)")
        enable_slack = st.checkbox("Slack Webhook 알림 사용", value=False)
        slack_webhook = st.text_input("Slack Webhook URL", type="password", value=get_secret_safe("SLACK_WEBHOOK_URL", ""))
        notify_watch = st.checkbox("WATCH(주의)도 외부 전송", value=False)

    # 알람에서만 예측을 사용(조기경보용)
    pred_df = pd.DataFrame(columns=["date", "pred"])
    with st.expander("조기경보를 위해 예측 생성(권장)", expanded=True):
        with st.spinner("예측 데이터 생성 중..."):
            fr = train_and_forecast_site(df_model, target, horizon=int(horizon), lags=14)
        if fr.mae is None or fr.pred_df.empty:
            st.info("예측 생성 불가 → 실측 기반 알람만 적용됩니다.")
        else:
            st.caption(f"백테스트 MAE: {fr.mae:.2f}")
            pred_df = fr.pred_df.copy()
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

    alert_state = evaluate_alert_state(
        df_site=df_model,
        target_col=target,
        alert_threshold=float(alert_threshold),
        sustain_days=int(sustain_days),
        anom_threshold=float(anom_threshold),
        delta_threshold=float(delta_threshold),
        earlywarn_days=int(earlywarn_days),
        forecast_df=pred_df if not pred_df.empty else None,
    )

    level = alert_state["level"]
    reasons = alert_state["reasons"]

    msg = f"[{selected_site}] {target} 상태: {level}"
    if reasons:
        msg += " / 사유: " + ", ".join(reasons)

    st.markdown("### 알람 상태")
    if level == "ALERT":
        st.error(msg)
        toast("경보(ALERT) 발생", icon="🚨")
    elif level == "WATCH":
        st.warning(msg)
        toast("주의(WATCH) 감지", icon="⚠️")
    else:
        st.success(msg)

    # 세션 로그(상태 변화 이벤트)
    if "alert_events" not in st.session_state:
        st.session_state["alert_events"] = []
    prev_level = st.session_state.get("prev_alert_level")

    if prev_level != level:
        st.session_state["prev_alert_level"] = level
        st.session_state["alert_events"].append(
            {
                "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "site": selected_site,
                "target": target,
                "level": level,
                "reasons": "; ".join(reasons) if reasons else "",
            }
        )

        # 외부 전송(상태 변화 시 1회)
        if enable_slack and slack_webhook:
            if level == "ALERT" or (notify_watch and level == "WATCH"):
                ok = send_slack_webhook(slack_webhook, msg)
                st.caption("Slack 전송: " + ("성공" if ok else "실패(웹훅/네트워크 확인)"))

    st.divider()
    st.markdown("### 알림 로그(세션)")
    events_df = pd.DataFrame(st.session_state["alert_events"])
    if len(events_df):
        st.dataframe(events_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("로그 초기화"):
                st.session_state["alert_events"] = []
                st.session_state["prev_alert_level"] = None
                st.rerun()
        with c2:
            csv_bytes = events_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="알림 로그 다운로드(CSV)",
                data=csv_bytes,
                file_name=f"alert_events_{selected_site[:40].replace(' ', '_')}.csv",
                mime="text/csv",
            )
    else:
        st.caption("상태 변화 이벤트가 아직 없습니다.")

    st.divider()
    st.markdown("### 경보 판정 테이블(최근 실측 + 예측)")
    recent_actual = df_model[["date", target]].dropna().sort_values("date").tail(14).copy()
    recent_actual["kind"] = "actual"
    recent_actual = recent_actual.rename(columns={target: "value"})

    future_forecast = pred_df.copy()
    if not future_forecast.empty:
        future_forecast["kind"] = "forecast"
        future_forecast = future_forecast.rename(columns={"pred": "value"})
    else:
        future_forecast = pd.DataFrame(columns=["date", "kind", "value"])

    log = pd.concat([recent_actual[["date", "kind", "value"]], future_forecast[["date", "kind", "value"]]], axis=0)
    log = log.sort_values("date").reset_index(drop=True)

    log["aqi_cat"] = log["value"].apply(aqi_category)
    log["sustain_count"] = sustained_flags(log["value"], float(alert_threshold)).astype(int)
    log["alert"] = np.where(log["sustain_count"] >= int(sustain_days), "ON", "OFF")

    st.dataframe(log.assign(date=log["date"].dt.date), use_container_width=True, hide_index=True)

            delta_color=delta_color
        )
