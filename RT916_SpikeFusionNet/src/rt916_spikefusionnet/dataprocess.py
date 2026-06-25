from chinese_calendar import is_workday, is_holiday
from borax.calendars import LunarDate
import pandas as pd
import datetime
import numpy as np


HISTORY_FORECAST_MAP = {
    "地方电厂总加实际值": "地方电厂总加预测值",
    "联络线受电负荷实际值": "联络线受电负荷预测值",
    "风电总加实际值": "风电总加预测值",
    "光伏总加实际值": "光伏总加预测值",
    "核电总加实际值": "核电总加预测值",
    "自备机组总加实际值": "自备机组总加预测值",
    "试验机组总加实际值": "试验机组总加预测值",
    "直调负荷实际值": "直调负荷预测值",
    "竞价空间实际值": "竞价空间预测值",
    "新能源总加实际值": "新能源总加预测值",
    "其他负荷总加": "其他负荷总加预测值",
    "总用电量": "总用电量预测值",
    "净负荷": "净负荷预测值",
    "新能源渗透率": "新能源渗透率预测值",
    "空间_新能源比": "空间_新能源比预测值",
}


def get_history_feature_name(feature_name):
    return HISTORY_FORECAST_MAP.get(feature_name, feature_name)


def split_excel_by_hours(df):
    df = df.copy()
    if "时刻" not in df.columns:
        print("错误: 数据中缺少'时刻'列")
        return None

    df["时刻"] = pd.to_datetime(df["时刻"])
    hours = df["时刻"].dt.hour

    mask_1_8 = hours.isin([1, 2, 3, 4, 5, 6, 7, 8])
    mask_9_16 = hours.isin([9, 10, 11, 12, 13, 14, 15, 16])
    mask_17_0 = hours.isin([17, 18, 19, 20, 21, 22, 23, 0])

    df_1_8 = df[mask_1_8].copy()
    df_9_16 = df[mask_9_16].copy()
    df_17_0 = df[mask_17_0].copy()

    total = len(df_1_8) + len(df_9_16) + len(df_17_0)
    if total != len(df):
        print(f"警告: 分割后总行数 ({total}) 与原始行数 ({len(df)}) 不一致")

    return df_1_8, df_9_16, df_17_0


def get_term_start_name(date_obj):
    return LunarDate.from_solar_date(date_obj.year, date_obj.month, date_obj.day).term


def find_initial_term(start_date):
    for i in range(1, 25):
        term = get_term_start_name(start_date - datetime.timedelta(days=i))
        if term:
            return term
    return "未知节气"


def adjust_date_for_0am(dt):
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return (dt - datetime.timedelta(days=1)).date()
    return dt.date()


def process_features(df):
    time_col = "时刻"
    if time_col not in df.columns:
        print(f"错误：列名 '{time_col}' 不在数据中")
        return None

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["TempDate"] = df[time_col].apply(adjust_date_for_0am)

    unique_dates = sorted(df["TempDate"].unique())
    current_term = get_term_start_name(unique_dates[0]) or find_initial_term(unique_dates[0])

    date_features_map = {}
    for d in unique_dates:
        term_on_day = get_term_start_name(d)
        if term_on_day:
            current_term = term_on_day
        date_features_map[d] = {
            "TempDate": d,
            "星期": d.weekday(),
            "节气名称": current_term,
            "是否上班": 1 if is_workday(d) else 0,
            "是否法定或周末休息": 1 if is_holiday(d) else 0,
        }

    feature_df = pd.DataFrame(list(date_features_map.values()))
    df_final = pd.merge(df, feature_df, on="TempDate", how="left")

    df_final["其他负荷总加"] = (
        df_final["地方电厂总加实际值"]
        + df_final["核电总加实际值"]
        + df_final["自备机组总加实际值"]
        + df_final["试验机组总加实际值"]
    )
    df_final["其他负荷总加预测值"] = (
        df_final["地方电厂总加预测值"]
        + df_final["核电总加预测值"]
        + df_final["自备机组总加预测值"]
        + df_final["试验机组总加预测值"]
    )

    df_final["总用电量"] = (
        df_final["直调负荷实际值"]
        + df_final["联络线受电负荷实际值"]
        + df_final["新能源总加实际值"]
        + df_final["其他负荷总加"]
    )
    df_final["总用电量预测值"] = (
        df_final["直调负荷预测值"]
        + df_final["联络线受电负荷预测值"]
        + df_final["新能源总加预测值"]
        + df_final["其他负荷总加预测值"]
    )

    df_final["净负荷"] = df_final["直调负荷实际值"] - df_final["新能源总加实际值"]
    df_final["净负荷预测值"] = df_final["直调负荷预测值"] - df_final["新能源总加预测值"]
    df_final["新能源渗透率"] = df_final["新能源总加实际值"] / (df_final["直调负荷实际值"] + 1e-5)
    df_final["新能源渗透率预测值"] = df_final["新能源总加预测值"] / (df_final["直调负荷预测值"] + 1e-5)
    df_final["空间_新能源比"] = df_final["竞价空间实际值"] / (df_final["新能源总加实际值"] + 1.0)
    df_final["空间_新能源比预测值"] = df_final["竞价空间预测值"] / (df_final["新能源总加预测值"] + 1.0)

    del df_final["TempDate"]
    return df_final


def feature_engineer_solar_terms(df):
    solar_terms_order = [
        "立春", "雨水", "惊蛰", "春分", "清明", "谷雨",
        "立夏", "小满", "芒种", "夏至", "小暑", "大暑",
        "立秋", "处暑", "白露", "秋分", "寒露", "霜降",
        "立冬", "小雪", "大雪", "冬至", "小寒", "大寒",
    ]
    solar_map = {term: i + 1 for i, term in enumerate(solar_terms_order)}

    if "节气名称" not in df.columns:
        raise ValueError("输入数据缺少 '节气名称' 列")

    df = df.copy()
    df["solar_term_ordinal"] = df["节气名称"].map(solar_map)

    na_count = df["solar_term_ordinal"].isna().sum()
    if na_count > 0:
        print(f"警告: 发现 {na_count} 行无法识别的节气名称")

    df["节气_sin"] = np.sin(2 * np.pi * (df["solar_term_ordinal"] - 1) / 24)
    df["节气_cos"] = np.cos(2 * np.pi * (df["solar_term_ordinal"] - 1) / 24)

    del df["solar_term_ordinal"]
    return df


def enrich_selected_features(df, target_col="实时电价"):
    """
    Add selected features for SpikeTimesNet:
    - time: hour, month, day_of_week
    - lag price: lag_48h, lag_168h, target_lag
    - ramps: ramp_load, ramp_solar (+ forecast-side ramp features)

    NOTE:
    lag features should be recomputed after asof cutoff in inference path
    to avoid future leakage.
    """
    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"])
    ts = out["时刻"]

    out["hour"] = ts.dt.hour + 1
    out["month"] = ts.dt.month
    out["day_of_week"] = ts.dt.dayofweek

    y = pd.to_numeric(out[target_col], errors="coerce")
    out["lag_48h"] = y.shift(48)
    out["lag_168h"] = y.shift(168)
    out["target_lag"] = np.where(out["day_of_week"] < 5, out["lag_168h"], out["lag_48h"])
    out["lag_48h"] = out["lag_48h"].ffill().fillna(0.0)
    out["lag_168h"] = out["lag_168h"].ffill().fillna(0.0)
    out["target_lag"] = pd.Series(out["target_lag"], index=out.index).ffill().fillna(0.0)

    load_pred = pd.to_numeric(out.get("直调负荷预测值"), errors="coerce")
    solar_pred = pd.to_numeric(out.get("新能源总加预测值"), errors="coerce")

    out["ramp_load"] = load_pred.diff().fillna(0.0)
    out["ramp_solar"] = solar_pred.diff().fillna(0.0)
    out["ramp_load_pred"] = load_pred.diff().fillna(0.0)
    out["ramp_solar_pred"] = solar_pred.diff().fillna(0.0)

    # DA-oriented stability features. These remain available to RT, but only the
    # DA branch will consume them through its own input factory.
    out["lag_24h"] = y.shift(24).ffill().fillna(0.0)
    out["lag_72h"] = y.shift(72).ffill().fillna(0.0)
    out["lag_336h"] = y.shift(336).ffill().fillna(0.0)
    out["target_lag_da"] = (
        0.60 * out["lag_24h"] + 0.25 * out["lag_168h"] + 0.15 * out["lag_336h"]
    )

    out["prevday_mean_target"] = y.shift(24).rolling(24, min_periods=1).mean().ffill().fillna(0.0)
    out["prevday_std_target"] = y.shift(24).rolling(24, min_periods=1).std().fillna(0.0)

    out["load_gap_prevday"] = (load_pred - load_pred.shift(24)).ffill().fillna(0.0)
    out["solar_gap_prevday"] = (solar_pred - solar_pred.shift(24)).ffill().fillna(0.0)
    out["net_load_gap_prevday"] = (
        pd.to_numeric(out.get("净负荷预测值"), errors="coerce")
        - pd.to_numeric(out.get("净负荷预测值"), errors="coerce").shift(24)
    ).ffill().fillna(0.0)

    out["load_pred_change_24h"] = load_pred.diff(24).fillna(0.0)
    out["solar_pred_change_24h"] = solar_pred.diff(24).fillna(0.0)
    return out


def recompute_target_dependent_selected_features(df, target_col="实时电价"):
    """
    Recompute only target-dependent lag and rolling features after a cutoff has
    already been applied to the target column.

    This keeps forecast-side and calendar features unchanged while ensuring that
    post-asof target-derived signals do not leak future truth.
    """
    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"])

    y = pd.to_numeric(out[target_col], errors="coerce")
    out["lag_48h"] = y.shift(48)
    out["lag_168h"] = y.shift(168)
    out["target_lag"] = np.where(out["day_of_week"] < 5, out["lag_168h"], out["lag_48h"])
    out["lag_48h"] = out["lag_48h"].ffill().fillna(0.0)
    out["lag_168h"] = out["lag_168h"].ffill().fillna(0.0)
    out["target_lag"] = pd.Series(out["target_lag"], index=out.index).ffill().fillna(0.0)

    out["lag_24h"] = y.shift(24).ffill().fillna(0.0)
    out["lag_72h"] = y.shift(72).ffill().fillna(0.0)
    out["lag_336h"] = y.shift(336).ffill().fillna(0.0)
    out["target_lag_da"] = (
        0.60 * out["lag_24h"] + 0.25 * out["lag_168h"] + 0.15 * out["lag_336h"]
    )

    out["prevday_mean_target"] = y.shift(24).rolling(24, min_periods=1).mean().ffill().fillna(0.0)
    out["prevday_std_target"] = y.shift(24).rolling(24, min_periods=1).std().fillna(0.0)
    return out


def enrich_period_local_features(df, target_col="实时电价", pred_len=8):
    """
    Add within-period daily anchors after the hour family split logic.
    This is especially useful for stage models such as 1-8 / 9-16 / 17-24,
    where "previous day same period" is not the same as a raw hourly shift.
    """
    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"])
    out = out.sort_values("时刻").reset_index(drop=True)

    y = pd.to_numeric(out[target_col], errors="coerce")
    out["period_lag_d1"] = y.shift(pred_len)
    out["period_lag_d2"] = y.shift(pred_len * 2)
    out["period_lag_w1"] = y.shift(pred_len * 7)
    out["period_prevday_mean"] = y.shift(pred_len).rolling(pred_len, min_periods=1).mean()
    out["period_prevday_std"] = y.shift(pred_len).rolling(pred_len, min_periods=1).std()
    out["period_mix_anchor"] = (
        0.50 * out["period_lag_d1"]
        + 0.30 * out["period_lag_w1"]
        + 0.20 * out["period_lag_d2"]
    )

    for col in [
        "period_lag_d1",
        "period_lag_d2",
        "period_lag_w1",
        "period_prevday_mean",
        "period_prevday_std",
        "period_mix_anchor",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").ffill().fillna(0.0)
    return out


# ==============================================================
#  DA→RT 联动特征注入
#  - 在 RT 训练/推理时,把"日前电价"作为代理"DA 模型输出"注入
#  - da_pred_today    : DA 模型对"今天"当前小时的预测 (=t 时刻的日前电价)
#  - da_pred_tomorrow : DA 模型对"明天"同一时刻的预测 (=t+24h 的日前电价)
#  - da_error_24h     : 过去 24h DA 实际(代理为预测)的平均绝对误差(滚动波动代理)
#  - da_ramp_seq_24h  : 过去 24h DA 的 ramp 序列特征
#  - da_pred_48h_lag  : 48h 前的 DA(给 1-8 点用的"昨天全天 DA"信号)
#  - da_pred_24h_lag  : 24h 前的 DA(给 9-16 点用的"今天 0-8 点 DA 实际"信号)
# ==============================================================
def enrich_da_linkage_features(
    df,
    da_pred_series=None,
    target_col="实时电价",
    da_col="日前电价",
    error_window=24,
):
    """
    在已包含 (时刻, 日前电价, ...) 的 df 上追加 6 个 DA 联动特征列.
    - da_pred_series: 可选,Series indexed by 时刻,作为"DA 模型预测"覆盖.
      若为 None,则把 df[da_col] 视作完美预测(训练/回测用).
    """
    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"])
    out = out.sort_values("时刻").reset_index(drop=True)

    if da_pred_series is not None:
        da_pred = pd.Series(da_pred_series).copy()
        da_pred.index = pd.to_datetime(da_pred.index)
        out = out.merge(
            da_pred.rename("da_model_pred").reset_index().rename(columns={"index": "时刻"}),
            on="时刻",
            how="left",
        )
        # 若某时刻缺预测,回退到日前电价(实际)
        out["da_model_pred"] = out["da_model_pred"].fillna(out[da_col])
    else:
        # 训练/回测代理:把"日前电价"本身视作"DA 模型输出"
        out["da_model_pred"] = pd.to_numeric(out[da_col], errors="coerce")

    # da_pred_today    : t 时刻的 DA 模型预测
    out["da_pred_today"] = pd.to_numeric(out["da_model_pred"], errors="coerce")
    # da_pred_tomorrow : t+24h 时刻的 DA 模型预测(向前看 24h)
    out["da_pred_tomorrow"] = out["da_pred_today"].shift(-24)
    # 24h 滞后:在 1-8 点场景时,看到的是"昨天全天 DA"
    out["da_pred_24h_lag"] = out["da_pred_today"].shift(24)
    out["da_pred_48h_lag"] = out["da_pred_today"].shift(48)
    # da_error_24h : 过去 24h DA ramp 的滚动平均绝对值(代理 DA 模型的 MAE 估计)
    da_diff = out["da_pred_today"].diff().abs()
    out["da_error_24h"] = da_diff.rolling(error_window, min_periods=1).mean()
    # da_ramp_seq_24h : DA 24h ramp 序列(取首/中/末三个位置作为紧凑表示)
    out["da_ramp_first_8"] = (
        out["da_pred_today"].rolling(8, min_periods=1).mean()
        - out["da_pred_today"].shift(8).rolling(8, min_periods=1).mean()
    )
    out["da_ramp_mid_8"] = (
        out["da_pred_today"].rolling(8, min_periods=1).mean()
        - out["da_pred_today"].shift(8)
    )
    out["da_ramp_pred_to_actual"] = out["da_pred_today"] - out["da_pred_24h_lag"]

    fill_cols = [
        "da_pred_today",
        "da_pred_tomorrow",
        "da_pred_24h_lag",
        "da_pred_48h_lag",
        "da_error_24h",
        "da_ramp_first_8",
        "da_ramp_mid_8",
        "da_ramp_pred_to_actual",
    ]
    for c in fill_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill().bfill().fillna(0.0)

    return out


def inject_external_da_predictions(df, external_da_df, da_col="日前电价", time_col="时刻"):
    """
    把外部 DA 模型(DataFrame 列:[时刻, 预测日前电价] 或 [时刻, da_pred])
    注入到 df 的"日前电价"列上(只覆盖原值,缺失部分保持原值).
    返回新 DataFrame(不修改原对象).
    """
    if external_da_df is None or len(external_da_df) == 0:
        return df
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    ext = external_da_df.copy()
    ext[time_col] = pd.to_datetime(ext[time_col])
    if "预测日前电价" in ext.columns:
        ext = ext.rename(columns={"预测日前电价": "_da_pred"})
    elif "da_pred" in ext.columns:
        ext = ext.rename(columns={"da_pred": "_da_pred"})
    elif da_col in ext.columns:
        ext = ext.rename(columns={da_col: "_da_pred"})
    else:
        return df
    ext = ext[[time_col, "_da_pred"]].drop_duplicates(subset=[time_col], keep="last")
    out = out.merge(ext, on=time_col, how="left")
    if da_col in out.columns:
        out[da_col] = out["_da_pred"].where(out["_da_pred"].notna(), out[da_col])
    else:
        out[da_col] = out["_da_pred"]
    out = out.drop(columns=["_da_pred"])
    return out

