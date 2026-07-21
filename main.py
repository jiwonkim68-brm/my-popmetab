import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="인구 대비 대사증후군 유병률 대시보드",
    page_icon="⚕️",
    layout="wide",
)

POP_FILE = "202606_202606_연령별인구현황_월간.csv"
POP_PREFIX = "2026년06월"

SIDO_NAME_MAP = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
    "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
}

RISK_FACTORS = ["복부비만", "높은혈압", "높은혈당", "고중성지방혈증", "낮은HDL콜레스테롤혈증"]


def show_data_caveat():
    st.warning(
        "⚠️ **이 대시보드의 수치는 '대사증후군 발생률'이 아닙니다.**\n\n"
        "2024년 국가건강검진(일반건강검진)을 받은 사람들 중, 검진 시점에 "
        "대사증후군 위험요인 5가지(복부비만·높은혈압·높은혈당·고중성지방혈증·"
        "낮은HDL콜레스테롤혈증) 중 **3개 이상을 가진 사람의 비율**이에요. "
        "새로 생긴 환자 수(발생)가 아니라, 그 시점에 그 상태였던 사람의 비율(유병)"
        "이라서, 이 대시보드에서는 '발생률' 대신 **'유병률(수검자 기준)'**이라는 "
        "표현을 씁니다."
    )


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_population():
    df = pd.read_csv(POP_FILE, encoding="cp949")
    for col in df.columns[1:]:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data
def load_metabolic_data():
    region_count = pd.read_csv("metabolic_region_count.csv")
    age_count = pd.read_csv("metabolic_age_count.csv")
    region_factor = pd.read_csv("metabolic_region_factor.csv")
    age_factor = pd.read_csv("metabolic_age_factor.csv")
    return region_count, age_count, region_factor, age_factor


with st.spinner("데이터를 불러오는 중이에요..."):
    df_pop_raw = load_population()
    df_region_count, df_age_count, df_region_factor, df_age_factor = load_metabolic_data()

st.title("⚕️ 인구 대비 대사증후군 유병률 대시보드")
st.caption(
    "행정안전부 주민등록 연령별 인구현황 + 2024년 건강검진통계연보 제7편(대사증후군)을 결합한 분석"
)
show_data_caveat()

# =========================================================
# 1단계. 인구 데이터에서 '시/도' 레벨(17개)만 뽑기
# =========================================================
df_pop_raw["code"] = df_pop_raw["행정구역"].str.extract(r"\((\d+)\)")[0].astype("Int64")
df_pop_raw["name_part"] = df_pop_raw["행정구역"].str.split("(").str[0].str.strip()

is_sido = df_pop_raw["name_part"].isin(SIDO_NAME_MAP.keys()) & (
    df_pop_raw["code"] % 100_000_000 == 0
)
df_sido_pop = df_pop_raw[is_sido].copy()
df_sido_pop["지역"] = df_sido_pop["name_part"].map(SIDO_NAME_MAP)

age_cols_total = [f"{POP_PREFIX}_계_{a}세" for a in range(100)] + [f"{POP_PREFIX}_계_100세 이상"]
elderly_cols = [f"{POP_PREFIX}_계_{a}세" for a in range(65, 100)] + [f"{POP_PREFIX}_계_100세 이상"]

df_sido_pop["총인구"] = df_sido_pop[f"{POP_PREFIX}_계_총인구수"]
df_sido_pop["65세이상인구"] = df_sido_pop[elderly_cols].sum(axis=1)
df_sido_pop["고령화율"] = df_sido_pop["65세이상인구"] / df_sido_pop["총인구"] * 100
df_sido_pop = df_sido_pop[["지역", "총인구", "65세이상인구", "고령화율"]]

# 전국 나이별(1세 단위) 인구 - 연령대별 근사 계산에 사용
df_sido_only = df_pop_raw[is_sido]
national_age_series = df_sido_only[age_cols_total].sum()
national_age_series.index = [
    100 if "100" in c else int(re.search(r"_(\d+)세", c).group(1))
    for c in national_age_series.index
]


def parse_age_bin(label: str):
    if "~" in label:
        lo_str, hi_str = label.replace("세", "").split("~")
        return int(lo_str), int(hi_str)
    m = re.search(r"\d+", label)
    return int(m.group()), 100


def population_for_bin(label: str) -> float:
    lo, hi = parse_age_bin(label)
    return float(national_age_series.loc[lo:hi].sum())


# =========================================================
# 지역/연령 데이터에 인구 붙이기 + 근사 지표 계산
# =========================================================
df_region = df_region_count.merge(df_sido_pop, on="지역", how="left")
df_region["인구_대비_유병률_근사"] = df_region["대사증후군소계_인원"] / df_region["총인구"] * 100

df_age = df_age_count.copy()
df_age["연령대_인구"] = df_age["연령구간"].apply(population_for_bin)
df_age["인구_대비_유병률_근사"] = df_age["대사증후군소계_인원"] / df_age["연령대_인구"] * 100
df_age["_정렬키"] = df_age["연령구간"].apply(lambda x: parse_age_bin(x)[0])
df_age = df_age.sort_values("_정렬키")
age_order = df_age["연령구간"].tolist()


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("📑 보기")
view_mode = st.sidebar.radio(
    "무엇을 볼까요?",
    ["지역별 유병률", "연령별 유병률 추이", "개별 위험요인 비교",
     "고령화율과의 상관관계", "시/도 × 연령 추정치 (모델)"],
)

st.divider()

# =========================================================
# 화면 1) 지역별 유병률
# =========================================================
if view_mode == "지역별 유병률":
    st.subheader("📍 시/도별 대사증후군 유병률")

    metric_choice = st.radio(
        "어떤 기준으로 볼까요?",
        ["수검자 기준 (통계연보 공식 비율)", "인구 대비 (근사치, 참고용)"],
        horizontal=True,
    )
    metric_col = (
        "대사증후군소계_비율"
        if metric_choice.startswith("수검자")
        else "인구_대비_유병률_근사"
    )

    df_plot = df_region.sort_values(metric_col, ascending=False)

    fig = go.Figure(
        go.Bar(
            x=df_plot["지역"], y=df_plot[metric_col],
            marker=dict(color=df_plot[metric_col], colorscale="Oranges"),
            text=[f"{v:.1f}%" for v in df_plot[metric_col]],
            textposition="outside",
            hovertemplate="%{x}<br>유병률: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="지역", yaxis_title="대사증후군 유병률 (%)",
        template="plotly_white", height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    if metric_col == "인구_대비_유병률_근사":
        st.caption(
            "💡 이 지표는 '전체 인구' 기준 근사치라, 검진을 적게 받은 지역은 "
            "실제보다 낮게 보일 수 있어요."
        )

    with st.expander("📋 지역별 상세 데이터 보기"):
        st.dataframe(
            df_plot[["지역", "총인구", "수검자수", "대사증후군소계_인원",
                     "대사증후군소계_비율", "인구_대비_유병률_근사"]]
            .rename(columns={"대사증후군소계_인원": "대사증후군_인원", "대사증후군소계_비율": "유병률(수검자기준,%)"})
            .style.format({
                "총인구": "{:,.0f}", "수검자수": "{:,.0f}", "대사증후군_인원": "{:,.0f}",
                "유병률(수검자기준,%)": "{:.1f}", "인구_대비_유병률_근사": "{:.1f}",
            }),
            use_container_width=True,
        )


# =========================================================
# 화면 2) 연령별 유병률 추이
# =========================================================
elif view_mode == "연령별 유병률 추이":
    st.subheader("📈 연령대별 대사증후군 유병률 추이")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_age["연령구간"], y=df_age["대사증후군소계_비율"],
            mode="lines+markers", name="유병률 (수검자 기준, 공식)",
            line=dict(width=3, color="#C0392B"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_age["연령구간"], y=df_age["주의군소계_비율"],
            mode="lines+markers", name="주의군 비율 (위험요인 1~2개)",
            line=dict(width=2, color="#F39C12", dash="dot"),
        )
    )
    fig.update_layout(
        xaxis_title="연령대", yaxis_title="비율 (%)",
        xaxis=dict(categoryorder="array", categoryarray=age_order),
        template="plotly_white", height=550, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 대사증후군 유병률이 몇 세부터 크게 높아지기 시작하는지 확인해보세요. "
        "다만 검진 대상 연령 기준이 연령대마다 다를 수 있어 해석에 주의가 필요해요."
    )

    with st.expander("📋 연령별 상세 데이터 보기"):
        st.dataframe(
            df_age[["연령구간", "연령대_인구", "수검자수", "대사증후군소계_인원",
                    "대사증후군소계_비율", "인구_대비_유병률_근사"]]
            .rename(columns={"대사증후군소계_인원": "대사증후군_인원", "대사증후군소계_비율": "유병률(수검자기준,%)"})
            .style.format({
                "연령대_인구": "{:,.0f}", "수검자수": "{:,.0f}", "대사증후군_인원": "{:,.0f}",
                "유병률(수검자기준,%)": "{:.1f}", "인구_대비_유병률_근사": "{:.1f}",
            }),
            use_container_width=True,
        )


# =========================================================
# 화면 3) 개별 위험요인 비교
# =========================================================
elif view_mode == "개별 위험요인 비교":
    st.subheader("🧪 개별 위험요인 5가지 비교")

    level = st.radio("기준을 선택하세요", ["지역별", "연령별"], horizontal=True)

    factor_ratio_cols = [f"{f}_비율" for f in RISK_FACTORS]

    if level == "지역별":
        fig = go.Figure()
        for factor in RISK_FACTORS:
            fig.add_trace(
                go.Bar(
                    x=df_region_factor["지역"], y=df_region_factor[f"{factor}_비율"],
                    name=factor,
                )
            )
        fig.update_layout(
            barmode="group", xaxis_title="지역", yaxis_title="보유율 (%)",
            template="plotly_white", height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        df_af = df_age_factor.copy()
        df_af["_정렬키"] = df_af["연령구간"].apply(lambda x: parse_age_bin(x)[0])
        df_af = df_af.sort_values("_정렬키")
        af_order = df_af["연령구간"].tolist()

        fig = go.Figure()
        for factor in RISK_FACTORS:
            fig.add_trace(
                go.Scatter(
                    x=df_af["연령구간"], y=df_af[f"{factor}_비율"],
                    mode="lines+markers", name=factor,
                )
            )
        fig.update_layout(
            xaxis_title="연령대", yaxis_title="보유율 (%)",
            xaxis=dict(categoryorder="array", categoryarray=af_order),
            template="plotly_white", height=550, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 5가지 위험요인 중 어떤 것이 지역/연령에 따라 가장 크게 차이 나는지 살펴보세요."
    )


# =========================================================
# 화면 4) 고령화율과의 상관관계
# =========================================================
elif view_mode == "고령화율과의 상관관계":
    st.subheader("🔗 지역 고령화율과 대사증후군 유병률의 관계")

    df_corr = df_region.dropna(subset=["고령화율", "대사증후군소계_비율"])
    corr_value = np.corrcoef(df_corr["고령화율"], df_corr["대사증후군소계_비율"])[0, 1]

    fig = go.Figure(
        go.Scatter(
            x=df_corr["고령화율"], y=df_corr["대사증후군소계_비율"],
            mode="markers+text", text=df_corr["지역"], textposition="top center",
            marker=dict(size=12, color="#16A085"),
        )
    )
    fig.update_layout(
        xaxis_title="고령화율 (65세 이상 인구 비율, %)",
        yaxis_title="대사증후군 유병률 (수검자 기준, %)",
        template="plotly_white", height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("피어슨 상관계수", f"{corr_value:.3f}")
    if abs(corr_value) >= 0.5:
        strength = "뚜렷한" if abs(corr_value) >= 0.7 else "어느 정도의"
        direction = "양(+)의" if corr_value > 0 else "음(-)의"
        st.info(f"고령화율과 유병률 사이에 {strength} {direction} 상관관계가 보여요.")
    else:
        st.info("고령화율과 유병률 사이에 뚜렷한 상관관계는 보이지 않아요.")

    st.caption(
        "💡 상관관계가 있다고 해서 '고령화가 대사증후군을 유발한다'는 인과관계를 "
        "의미하지는 않아요. 지역별 생활습관, 검진 참여율 등 다른 요인도 함께 작용할 수 있어요."
    )


# =========================================================
# 화면 5) 시/도 × 연령 추정치 (모델) — 근사 추정, 실측값 아님!
# =========================================================
else:
    st.subheader("🧮 시/도 × 연령 대사증후군 유병률 — 추정치 (모델)")

    st.error(
        "🚨 **이 화면의 숫자는 실제로 관측된 값이 아니라, '추정치'입니다.**\n\n"
        "국민건강보험공단 통계연보에는 '지역별' 유병률과 '연령별' 유병률이 "
        "각각 따로만 공개되어 있고, 두 기준을 동시에 교차한 표는 없어요. "
        "그래서 아래 두 가지를 곱하는 단순한 모형으로 추정했습니다.\n\n"
        "- 지역 배율 = 그 지역의 유병률 ÷ 전국 평균 유병률\n"
        "- 연령 배율 = 그 연령대의 유병률 ÷ 전국 평균 유병률\n"
        "- **추정 유병률 = 전국 평균 × 지역 배율 × 연령 배율**\n\n"
        "이 모형은 '지역 효과'와 '연령 효과'가 서로 독립적으로 작용한다고 "
        "가정해요. 하지만 실제로는 지역마다 연령 구성이 다르고, 지역 고유의 "
        "생활습관·환경 요인이 연령대별로 다르게 작용할 수 있어서 — 특히 "
        "고령 인구 비율이 아주 높거나 낮은 지역에서는 이 추정이 실제와 "
        "차이가 클 수 있습니다."
    )

    # --- 전국 평균 유병률(P) 계산 ---
    national_rate = (
        df_region_count["대사증후군소계_인원"].sum()
        / df_region_count["수검자수"].sum()
        * 100
    )

    region_multiplier = df_region_count.set_index("지역")["대사증후군소계_비율"] / national_rate
    age_multiplier = df_age.set_index("연령구간")["대사증후군소계_비율"] / national_rate
    age_multiplier = age_multiplier.reindex(age_order)  # 나이 순서로 정렬

    # --- 17(지역) x 14(연령) 추정치 행렬 만들기 ---
    estimate_matrix = np.outer(region_multiplier.values, age_multiplier.values) * national_rate
    region_list = region_multiplier.index.tolist()

    fig_heat = go.Figure(
        go.Heatmap(
            z=estimate_matrix,
            x=age_order,
            y=region_list,
            colorscale="YlOrRd",
            colorbar=dict(title="추정 유병률(%)"),
            hovertemplate="%{y} · %{x}<br>추정 유병률: %{z:.1f}%<extra></extra>",
        )
    )
    fig_heat.update_layout(
        title="시/도 × 연령대 대사증후군 유병률 추정치 (⚠️ 모델 기반 추정값)",
        xaxis_title="연령대", yaxis_title="지역",
        template="plotly_white", height=650,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # --- 지역 하나를 골라서, 그 지역의 연령별 추정 곡선 보기 ---
    st.subheader("📍 지역별 연령 곡선 (추정치)")
    picked_region = st.selectbox("지역을 선택하세요", region_list)

    region_curve = age_multiplier.values * region_multiplier[picked_region] * national_rate
    national_curve = age_multiplier.values * national_rate  # 전국 평균 곡선 (비교용)

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=age_order, y=region_curve, mode="lines+markers",
            name=f"{picked_region} (추정치)", line=dict(width=3, color="#E67E22"),
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=age_order, y=national_curve, mode="lines+markers",
            name="전국 평균 (실측)", line=dict(width=2, color="#7F8C8D", dash="dash"),
        )
    )
    fig_line.update_layout(
        xaxis_title="연령대", yaxis_title="유병률 (%)",
        xaxis=dict(categoryorder="array", categoryarray=age_order),
        template="plotly_white", height=500, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.caption(
        f"💡 {picked_region}의 지역 배율은 {region_multiplier[picked_region]:.2f}배예요 "
        f"(전국 평균 대비). 이 배율을 전국 연령별 곡선에 곱해서 추정한 곡선이라, "
        "실제 그 지역의 연령별 곡선과는 다를 수 있어요."
    )

    with st.expander("📋 추정치 표 전체 보기"):
        df_matrix = pd.DataFrame(estimate_matrix, index=region_list, columns=age_order)
        st.dataframe(df_matrix.style.format("{:.1f}"), use_container_width=True)

st.divider()
st.caption(
    "💡 데이터 출처: 행정안전부 주민등록 연령별 인구현황(2026.06) · "
    "국민건강보험공단 2024년 건강검진통계연보 제7편(대사증후군) · "
    "이 앱은 학습·연습 목적으로 만들어졌어요."
)
