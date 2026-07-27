# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from tqdm.auto import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 参数区
# ============================================================
DEM_CSV = "陕甘八县的高程数据.csv"
ATT2_XLSX = "附件2  秦直道及周边地形和相关遗迹的数据.xlsx"

RANDOM_SEED = 42

SAVE_FIG = True
FIG_DPI = 150
OVERVIEW_DOWNSAMPLE = 8

# -------- 背景点匹配参数 --------
TARGET_BG_N = 4000
BG_MAX_ROUNDS = 40
BG_PER_ROUND = 150000

RIDGE1_BG_MAX_DIST = 1200.0
ROAD_BG_MIN_DIST = 300.0
ROAD_BG_MAX_DIST = 5000.0

MATCH_BINS = 20

# -------- 特征 --------
ROUTE_COMPARE_FEATURES = [
    "elevation",
    "slope_deg",
    "aspect_deg",
    "local_relief_3x3",
    "tpi_3x3",
    "roughness_3x3",
    "dist_to_ridge1",
    "dist_to_ridge2",
    "dist_to_river",
    "ridge1_relative_elev",
]

RELATION_FEATURES = [
    "elevation",
    "slope_deg",
    "local_relief_3x3",
    "tpi_3x3",
    "roughness_3x3",
    "dist_to_ridge1",
    "dist_to_ridge2",
    "dist_to_river",
    "ridge1_relative_elev",
    "angle_diff_to_ridge1",
]

SITE_FEATURES = [
    "elevation",
    "slope_deg",
    "aspect_deg",
    "local_relief_3x3",
    "tpi_3x3",
    "roughness_3x3",
    "dist_to_ridge1",
    "dist_to_ridge2",
    "dist_to_river",
    "dist_to_road",
    "ridge1_relative_elev",
]

assert os.path.exists(DEM_CSV), f"未找到文件: {DEM_CSV}"
assert os.path.exists(ATT2_XLSX), f"未找到文件: {ATT2_XLSX}"
print("文件检查通过。")


# ============================================================
# 1. 工具函数
# ============================================================
def standardize_xy(df):
    cols = list(df.columns)
    xcol = None
    ycol = None

    exact_x = ["x", "x坐标", "x坐标/m", "x坐标（m）", "x坐标(m)"]
    exact_y = ["y", "y坐标", "y坐标/m", "y坐标（m）", "y坐标(m)"]

    for c in cols:
        sc = str(c).strip().lower()
        if sc in [s.lower() for s in exact_x]:
            xcol = c
        if sc in [s.lower() for s in exact_y]:
            ycol = c

    if xcol is None:
        xs = [c for c in cols if "x" in str(c).lower()]
        if len(xs) == 0:
            raise ValueError(f"无法识别 x 列，列名：{cols}")
        xcol = xs[0]

    if ycol is None:
        ys = [c for c in cols if "y" in str(c).lower()]
        if len(ys) == 0:
            raise ValueError(f"无法识别 y 列，列名：{cols}")
        ycol = ys[0]

    return df.copy().rename(columns={xcol: "x", ycol: "y"})


def angle_diff_deg(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def compute_line_angles(df_xy):
    xy = df_xy[["x", "y"]].to_numpy(dtype=float)
    n = len(xy)
    if n < 2:
        return np.full(n, np.nan)

    ang = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            dx_ = xy[i + 1, 0] - xy[i, 0]
            dy_ = xy[i + 1, 1] - xy[i, 1]
        elif i == n - 1:
            dx_ = xy[i, 0] - xy[i - 1, 0]
            dy_ = xy[i, 1] - xy[i - 1, 1]
        else:
            dx_ = xy[i + 1, 0] - xy[i - 1, 0]
            dy_ = xy[i + 1, 1] - xy[i - 1, 1]

        a = np.degrees(np.arctan2(dy_, dx_))
        if a < 0:
            a += 360
        ang[i] = a
    return ang


def split_polyline_by_seq(df):
    """
    二级分水岭、河网：每段内部序号均从1开始
    因此按“序号回跳或重置”切段
    """
    if "序号" not in df.columns:
        return [df.copy().reset_index(drop=True)]

    seq = pd.to_numeric(df["序号"], errors="coerce").to_numpy()
    breaks = [0]
    for i in range(1, len(seq)):
        if np.isnan(seq[i - 1]) or np.isnan(seq[i]):
            continue
        if seq[i] <= seq[i - 1]:
            breaks.append(i)
    breaks.append(len(df))

    parts = []
    for i in range(len(breaks) - 1):
        sub = df.iloc[breaks[i]:breaks[i + 1]].copy().reset_index(drop=True)
        if len(sub) >= 2:
            parts.append(sub)

    if len(parts) == 0:
        parts = [df.copy().reset_index(drop=True)]
    return parts


def build_segment_index_single(df, name="单段线"):
    arr = df[["x", "y"]].to_numpy(dtype=float)
    seg_geoms = []
    seg_angles = []

    for i in tqdm(range(len(arr) - 1), desc=f"建立{name}线段"):
        p1 = arr[i]
        p2 = arr[i + 1]
        if np.array_equal(p1, p2):
            continue

        seg_geoms.append(LineString([tuple(p1), tuple(p2)]))
        dx_ = p2[0] - p1[0]
        dy_ = p2[1] - p1[1]
        ang = np.degrees(np.arctan2(dy_, dx_))
        if ang < 0:
            ang += 360
        seg_angles.append(ang)

    tree = STRtree(seg_geoms)
    return tree, seg_geoms, np.array(seg_angles, dtype=float)


def build_segment_index_multi(df, name="多段线"):
    parts = split_polyline_by_seq(df)
    seg_geoms = []
    seg_angles = []

    for part in tqdm(parts, desc=f"建立{name}线段索引"):
        arr = part[["x", "y"]].to_numpy(dtype=float)
        for i in range(len(arr) - 1):
            p1 = arr[i]
            p2 = arr[i + 1]
            if np.array_equal(p1, p2):
                continue

            seg_geoms.append(LineString([tuple(p1), tuple(p2)]))
            dx_ = p2[0] - p1[0]
            dy_ = p2[1] - p1[1]
            ang = np.degrees(np.arctan2(dy_, dx_))
            if ang < 0:
                ang += 360
            seg_angles.append(ang)

    tree = STRtree(seg_geoms)
    return tree, seg_geoms, np.array(seg_angles, dtype=float)


def query_nearest_segment_distance(tree, xs, ys, chunk_size=5000, desc="最近线段距离"):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.empty(len(xs), dtype=float)

    n_chunks = (len(xs) + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, len(xs), chunk_size), desc=desc, total=n_chunks):
        end = min(start + chunk_size, len(xs))
        pts = np.array([Point(x, y) for x, y in zip(xs[start:end], ys[start:end])], dtype=object)

        res = tree.query_nearest(pts, return_distance=True, all_matches=False)
        _, dists = res
        dists = np.asarray(dists, dtype=float)

        if len(dists) != (end - start):
            sub = np.empty(end - start, dtype=float)
            for k, pt in enumerate(pts):
                one_res = tree.query_nearest([pt], return_distance=True, all_matches=False)
                _, one_dist = one_res
                one_dist = np.asarray(one_dist, dtype=float)
                sub[k] = one_dist[0] if len(one_dist) > 0 else np.nan
            out[start:end] = sub
        else:
            out[start:end] = dists

    return out


def query_nearest_segment_angle(tree, seg_angles, xs, ys, chunk_size=5000, desc="最近线段方向"):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    out = np.empty(len(xs), dtype=float)

    n_chunks = (len(xs) + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, len(xs), chunk_size), desc=desc, total=n_chunks):
        end = min(start + chunk_size, len(xs))
        pts = np.array([Point(x, y) for x, y in zip(xs[start:end], ys[start:end])], dtype=object)
        idxs = tree.query_nearest(pts, return_distance=False, all_matches=False)
        seg_idx = idxs[1]
        out[start:end] = seg_angles[seg_idx]

    return out


def nearest_point_distance(xs, ys, point_xy_df, desc="最近点距离", chunk_size=3000):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if len(point_xy_df) == 0:
        return np.full(len(xs), np.nan)

    tree = cKDTree(point_xy_df[["x", "y"]].to_numpy(dtype=float))
    out = np.empty(len(xs), dtype=float)

    n_chunks = (len(xs) + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, len(xs), chunk_size), desc=desc, total=n_chunks):
        end = min(start + chunk_size, len(xs))
        pts = np.column_stack([xs[start:end], ys[start:end]])
        dist, _ = tree.query(pts, k=1)
        out[start:end] = dist

    return out


def sample_interp(interp_obj, xs, ys):
    pts = np.column_stack([np.atleast_1d(ys), np.atleast_1d(xs)])
    return interp_obj(pts)


def nearest_grid_index(arr, vals):
    vals = np.asarray(vals)
    idx = np.searchsorted(arr, vals)
    idx = np.clip(idx, 1, len(arr) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(arr[right] - vals) < np.abs(arr[left] - vals)
    return np.where(choose_right, right, left)


def local_window_features(z, x_coords, y_coords, xs, ys, desc="3x3局部窗口特征"):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    ix_all = nearest_grid_index(x_coords, xs)
    iy_all = nearest_grid_index(y_coords, ys)

    relief = np.empty(len(xs), dtype=float)
    tpi = np.empty(len(xs), dtype=float)
    rough = np.empty(len(xs), dtype=float)

    for i in tqdm(range(len(xs)), desc=desc):
        ix = ix_all[i]
        iy = iy_all[i]

        r0 = max(0, iy - 1)
        r1 = min(z.shape[0], iy + 2)
        c0 = max(0, ix - 1)
        c1 = min(z.shape[1], ix + 2)

        win = z[r0:r1, c0:c1].copy()
        center = z[iy, ix]

        vals = win[~np.isnan(win)]
        if len(vals) == 0 or np.isnan(center):
            relief[i] = np.nan
            tpi[i] = np.nan
            rough[i] = np.nan
            continue

        relief[i] = np.max(vals) - np.min(vals)
        rough[i] = np.std(vals)

        rr = iy - r0
        cc = ix - c0
        if 0 <= rr < win.shape[0] and 0 <= cc < win.shape[1]:
            win[rr, cc] = np.nan

        neigh = win[~np.isnan(win)]
        if len(neigh) == 0:
            tpi[i] = np.nan
        else:
            tpi[i] = center - np.mean(neigh)

    return relief, tpi, rough


def check_points_status_by_interp(xs, ys, xmin, xmax, ymin, ymax, interp_elev):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    in_extent = (
        (xs >= xmin) & (xs <= xmax) &
        (ys >= ymin) & (ys <= ymax)
    )

    elev = sample_interp(interp_elev, xs, ys)
    in_valid = np.isfinite(elev)
    return in_extent, in_valid


def classify_sites(sites_df):
    tp = sites_df["类型"].astype(str).str.strip()
    beacon = sites_df[tp.str.contains("烽火台", na=False)].copy()
    pass_site = sites_df[tp.str.contains("关隘", na=False)].copy()
    other_sites = sites_df[tp.str.contains("相关遗存", na=False)].copy()
    return beacon, pass_site, other_sites


def summarize_describe(df, cols, name="样本"):
    out = df[cols].describe().T.reset_index().rename(columns={"index": "feature"})
    out.insert(0, "sample_name", name)
    return out


def summarize_tests(df_a, df_b, feature_cols, name_a="A", name_b="B"):
    rows = []
    for col in feature_cols:
        x = df_a[col].replace([np.inf, -np.inf], np.nan).dropna()
        y = df_b[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) < 3 or len(y) < 3:
            rows.append({
                "feature": col,
                f"{name_a}_mean": np.nan,
                f"{name_b}_mean": np.nan,
                f"{name_a}_median": np.nan,
                f"{name_b}_median": np.nan,
                "median_gap": np.nan,
                "p_value_mannwhitney": np.nan,
                "effect_direction": "NA"
            })
            continue

        try:
            _, p = mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            p = np.nan

        xa = x.median()
        yb = y.median()

        if xa > yb:
            direction = f"{name_a}_higher"
        elif xa < yb:
            direction = f"{name_a}_lower"
        else:
            direction = "equal"

        rows.append({
            "feature": col,
            f"{name_a}_mean": x.mean(),
            f"{name_b}_mean": y.mean(),
            f"{name_a}_median": xa,
            f"{name_b}_median": yb,
            "median_gap": xa - yb,
            "p_value_mannwhitney": p,
            "effect_direction": direction
        })

    out = pd.DataFrame(rows)
    out["abs_median_gap"] = np.abs(out["median_gap"])
    out = out.sort_values(["p_value_mannwhitney", "abs_median_gap"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_road_line_and_chainage(road_df):
    arr = road_df[["x", "y"]].to_numpy(dtype=float)
    seg_lengths = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    line = LineString(arr)
    return line, cum


def project_points_to_road_chainage(xs, ys, road_line):
    chainages = []
    nearest_xy = []

    for x, y in zip(xs, ys):
        p = Point(float(x), float(y))
        s = road_line.project(p)
        q = road_line.interpolate(s)
        chainages.append(s)
        nearest_xy.append([q.x, q.y])

    chainages = np.asarray(chainages, dtype=float)
    nearest_xy = np.asarray(nearest_xy, dtype=float).reshape(-1, 2) if len(nearest_xy) > 0 else np.empty((0, 2), dtype=float)
    return chainages, nearest_xy


def add_road_projection_info(df, road_line):
    out = df.copy()

    if len(out) == 0:
        out["road_chainage_m"] = np.nan
        out["nearest_road_x"] = np.nan
        out["nearest_road_y"] = np.nan
        return out

    ch, qxy = project_points_to_road_chainage(
        out["x"].to_numpy(dtype=float),
        out["y"].to_numpy(dtype=float),
        road_line
    )

    out["road_chainage_m"] = ch
    if len(qxy) == 0:
        out["nearest_road_x"] = np.nan
        out["nearest_road_y"] = np.nan
    else:
        out["nearest_road_x"] = qxy[:, 0]
        out["nearest_road_y"] = qxy[:, 1]

    return out


def make_dist_bin(series, bins=20):
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < bins:
        bins = max(5, min(len(s), bins))
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(s, qs))
    if len(edges) < 3:
        edges = np.array([s.min(), s.median(), s.max() + 1e-9])
    return edges


def match_background_by_ridge_dist(road_df, bg_cand, target_n=4000, bins=20, random_state=42):
    rng = np.random.default_rng(random_state)

    road_use = road_df.copy()
    bg_use = bg_cand.copy()

    road_use = road_use[np.isfinite(road_use["dist_to_ridge1"])].copy()
    bg_use = bg_use[np.isfinite(bg_use["dist_to_ridge1"])].copy()

    if len(road_use) == 0 or len(bg_use) == 0:
        return bg_use.head(min(target_n, len(bg_use))).copy()

    edges = make_dist_bin(road_use["dist_to_ridge1"], bins=bins)

    road_use["ridge_bin"] = pd.cut(road_use["dist_to_ridge1"], bins=edges, include_lowest=True, duplicates="drop")
    bg_use["ridge_bin"] = pd.cut(bg_use["dist_to_ridge1"], bins=edges, include_lowest=True, duplicates="drop")

    road_counts = road_use["ridge_bin"].value_counts(dropna=False).sort_index()
    bg_groups = {k: v for k, v in bg_use.groupby("ridge_bin")}

    selected = []
    total_road = road_counts.sum()

    for bin_key, cnt in road_counts.items():
        if pd.isna(bin_key):
            continue
        if bin_key not in bg_groups:
            continue

        group = bg_groups[bin_key]
        if len(group) == 0:
            continue

        take = int(round(target_n * cnt / total_road))
        take = max(1, take)

        if len(group) <= take:
            selected.append(group.copy())
        else:
            idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
            selected.append(group.loc[idx].copy())

    if len(selected) == 0:
        return bg_use.head(min(target_n, len(bg_use))).copy()

    out = pd.concat(selected, ignore_index=False).drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    if len(out) > target_n:
        out = out.sample(target_n, random_state=random_state).reset_index(drop=True)

    return out


def compute_vif_table(df, feature_cols):
    data = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(data)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rows = []
    p = X.shape[1]
    for i in range(p):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)

        if X_other.shape[1] == 0:
            vif = 1.0
        else:
            X_design = np.column_stack([np.ones(len(X_other)), X_other])
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            y_hat = X_design @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 0 if ss_tot == 0 else 1 - ss_res / ss_tot
            vif = np.inf if (1 - r2) <= 1e-12 else 1 / (1 - r2)

        rows.append({"feature": feature_cols[i], "VIF": vif})

    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def compute_correlations(df, feature_cols):
    data = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    return pearson, spearman


def compute_pca(df, feature_cols, n_components=5):
    data = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(data)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_components = min(n_components, Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    scores = pca.fit_transform(Xs)

    loading = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    ).reset_index().rename(columns={"index": "feature"})

    explained = pd.DataFrame({
        "主成分": [f"PC{i+1}" for i in range(pca.n_components_)],
        "解释方差比": pca.explained_variance_ratio_,
        "累计解释方差比": np.cumsum(pca.explained_variance_ratio_)
    })

    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return loading, explained, scores_df


def stratified_route_summary_by_ridge(road_df, bins=5):
    df = road_df.copy()
    use = df["dist_to_ridge1"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < bins:
        bins = max(3, min(len(use), bins))
    edges = np.unique(np.quantile(use, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame()

    df["ridge_bin"] = pd.cut(df["dist_to_ridge1"], bins=edges, include_lowest=True, duplicates="drop")
    out = df.groupby("ridge_bin").agg(
        样本数=("dist_to_ridge1", "size"),
        一级分水岭距离中位数=("dist_to_ridge1", "median"),
        坡度中位数=("slope_deg", "median"),
        局部起伏度中位数=("local_relief_3x3", "median"),
        粗糙度中位数=("roughness_3x3", "median"),
        河网距离中位数=("dist_to_river", "median"),
        相对一级分水岭高差中位数=("ridge1_relative_elev", "median"),
    ).reset_index()
    return out


def build_principle_summary(road_df, bg_feat, beacon_feat, pass_feat):
    rows = []

    if len(road_df) > 0 and len(bg_feat) > 0:
        rows.append({
            "类别": "路线规划原则",
            "原则名称": "贴近一级分水岭布线",
            "量化证据": f"道路点到一级分水岭距离中位数={road_df['dist_to_ridge1'].median():.2f} m；背景点中位数={bg_feat['dist_to_ridge1'].median():.2f} m",
            "结论": "秦直道优先沿一级分水岭附近展布"
        })
        rows.append({
            "类别": "路线规划原则",
            "原则名称": "优先选择缓坡带",
            "量化证据": f"道路点坡度中位数={road_df['slope_deg'].median():.2f}°；背景点中位数={bg_feat['slope_deg'].median():.2f}°",
            "结论": "路线总体倾向于选择坡度较缓的通行带"
        })
        rows.append({
            "类别": "路线规划原则",
            "原则名称": "优先选择低起伏带",
            "量化证据": f"道路点局部起伏度中位数={road_df['local_relief_3x3'].median():.2f} m；背景点中位数={bg_feat['local_relief_3x3'].median():.2f} m",
            "结论": "路线倾向于避开局部剧烈起伏地段"
        })
        rows.append({
            "类别": "路线规划原则",
            "原则名称": "减少河谷切割",
            "量化证据": f"道路点到河网距离中位数={road_df['dist_to_river'].median():.2f} m；背景点中位数={bg_feat['dist_to_river'].median():.2f} m",
            "结论": "主线总体避免频繁进入河谷和密集水系地段"
        })
        rows.append({
            "类别": "路线规划原则",
            "原则名称": "方向顺应地形骨架",
            "量化证据": f"道路方向与一级分水岭方向夹角中位数={road_df['angle_diff_to_ridge1'].median():.2f}°",
            "结论": "路线走向与主地形骨架具有较强一致性"
        })

    if len(beacon_feat) > 0:
        rows.append({
            "类别": "设施设置原则",
            "原则名称": "烽火台设于较高、较开阔位置",
            "量化证据": f"烽火台高程中位数={beacon_feat['elevation'].median():.2f} m；相对一级分水岭高差中位数={beacon_feat['ridge1_relative_elev'].median():.2f} m",
            "结论": "烽火台偏向占据较高势位置，以增强瞭望和传讯能力"
        })
        rows.append({
            "类别": "设施设置原则",
            "原则名称": "烽火台靠近主路线但不与道路重合",
            "量化证据": f"烽火台到道路距离中位数={beacon_feat['dist_to_road'].median():.2f} m",
            "结论": "烽火台通常依托主线附近制高点设置，以兼顾监视与联络"
        })

    if len(pass_feat) > 0:
        rows.append({
            "类别": "设施设置原则",
            "原则名称": "关隘紧贴交通控制节点",
            "量化证据": f"关隘到道路距离中位数={pass_feat['dist_to_road'].median():.2f} m",
            "结论": "关隘主要设置在交通线控制节点上"
        })
        rows.append({
            "类别": "设施设置原则",
            "原则名称": "关隘位于可扼守的地形位置",
            "量化证据": f"关隘到一级分水岭距离中位数={pass_feat['dist_to_ridge1'].median():.2f} m；到河网距离中位数={pass_feat['dist_to_river'].median():.2f} m",
            "结论": "关隘多设于路线与地形约束共同作用的瓶颈地段"
        })

    return pd.DataFrame(rows)


def plot_corr_heatmap(corr_df, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_pca_scatter(scores_df, save_path):
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(scores_df["PC1"], scores_df["PC2"], s=10, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("道路点 PCA 前两主成分分布")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


# ============================================================
# 2. 读取 DEM（修正版：强制翻转 z）
# ============================================================
dem_raw = pd.read_csv(DEM_CSV, header=None)
print("\n[DEM CSV 原始形状]", dem_raw.shape)

x_coords = pd.to_numeric(dem_raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
y_coords = pd.to_numeric(dem_raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
z = dem_raw.iloc[1:, 1:].replace("NA", np.nan)
z = z.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

print("\n[DEM 原始解析后]")
print("x 数量:", len(x_coords))
print("y 数量:", len(y_coords))
print("高程矩阵形状:", z.shape)
print("高程最小值(忽略 nan):", np.nanmin(z))
print("高程最大值(忽略 nan):", np.nanmax(z))

if not np.all(np.diff(x_coords) > 0):
    raise ValueError("x_coords 不是严格递增，无法安全构建插值器。")
if not np.all(np.diff(y_coords) > 0):
    raise ValueError("y_coords 不是严格递增，请检查 CSV。")

# 关键修正：只翻转 z，不翻转 y_coords
z = z[::-1, :]

dx = float(np.median(np.diff(x_coords)))
dy = float(np.median(np.diff(y_coords)))

xmin, xmax = x_coords.min(), x_coords.max()
ymin, ymax = y_coords.min(), y_coords.max()

print("\n[DEM 修正后]")
print("已对高程矩阵 z 做上下翻转：z = z[::-1, :]")
print("dx =", dx)
print("dy =", dy)
print("x范围:", xmin, "~", xmax)
print("y范围:", ymin, "~", ymax)


# ============================================================
# 3. 读取附件2
# ============================================================
xls = pd.ExcelFile(ATT2_XLSX)
print("\n[附件2 sheet 名称]")
print(xls.sheet_names)

road = pd.read_excel(ATT2_XLSX, sheet_name="秦直道")
ridge1 = pd.read_excel(ATT2_XLSX, sheet_name="一级分水岭")
ridge2 = pd.read_excel(ATT2_XLSX, sheet_name="二级分水岭")
river = pd.read_excel(ATT2_XLSX, sheet_name="河网")
sites = pd.read_excel(ATT2_XLSX, sheet_name="烽火台、关隘及相关遗存")

road = standardize_xy(road)
ridge1 = standardize_xy(ridge1)
ridge2 = standardize_xy(ridge2)
river = standardize_xy(river)
sites = standardize_xy(sites)

print("\n[附件2各表规模]")
print("秦直道:", road.shape)
print("一级分水岭:", ridge1.shape)
print("二级分水岭:", ridge2.shape)
print("河网:", river.shape)
print("遗迹点:", sites.shape)

beacon, pass_site, other_sites = classify_sites(sites)

print("\n[遗迹类型统计]")
print(sites["类型"].value_counts(dropna=False))


# ============================================================
# 4. 建立线段索引与路线对象
# ============================================================
print("\n[建立线段索引中...]")
road_tree, road_segs, road_seg_angles = build_segment_index_single(road, name="秦直道")
ridge1_tree, ridge1_segs, ridge1_seg_angles = build_segment_index_single(ridge1, name="一级分水岭")
ridge2_tree, ridge2_segs, ridge2_seg_angles = build_segment_index_multi(ridge2, name="二级分水岭")
river_tree, river_segs, river_seg_angles = build_segment_index_multi(river, name="河网")

road_line, road_cum = build_road_line_and_chainage(road)

print("秦直道线段数:", len(road_segs))
print("一级分水岭线段数:", len(ridge1_segs))
print("二级分水岭线段数:", len(ridge2_segs))
print("河网线段数:", len(river_segs))

if len(road_segs) == 0 or len(ridge1_segs) == 0:
    raise RuntimeError("建线段失败，请检查附件2数据。")


# ============================================================
# 5. 总览图
# ============================================================
if SAVE_FIG:
    print("\n[生成总览图中...]")
    step = max(1, int(OVERVIEW_DOWNSAMPLE))
    z_plot = z[::step, ::step]

    plt.figure(figsize=(12, 9))
    plt.imshow(
        z_plot,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="auto",
        cmap="terrain"
    )
    plt.colorbar(label="Elevation (m)")
    plt.plot(road["x"], road["y"], color="yellow", linewidth=1.3, label="秦直道")
    plt.plot(ridge1["x"], ridge1["y"], color="white", linewidth=1.0, label="一级分水岭")
    plt.scatter(ridge2["x"].iloc[::3], ridge2["y"].iloc[::3], s=5, color="cyan", alpha=0.65, label="二级分水岭")
    plt.scatter(river["x"].iloc[::8], river["y"].iloc[::8], s=3, color="blue", alpha=0.30, label="河网")

    if len(beacon) > 0:
        plt.scatter(beacon["x"], beacon["y"], s=70, color="orange", label="烽火台")
    if len(pass_site) > 0:
        plt.scatter(pass_site["x"], pass_site["y"], s=80, color="#1f77b4", label="关隘")
    if len(other_sites) > 0:
        plt.scatter(other_sites["x"], other_sites["y"], s=40, color="limegreen", label="相关遗存")

    plt.title("问题2：DEM + 秦直道 + 分水岭 + 河网 + 遗迹")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("问题2_总览图.png", dpi=FIG_DPI)
    plt.close()
    print("已保存: 问题2_总览图.png")


# ============================================================
# 6. DEM 插值器、坡度、坡向
# ============================================================
interp_elev = RegularGridInterpolator(
    (y_coords, x_coords), z,
    method="linear",
    bounds_error=False,
    fill_value=np.nan
)

print("\n[计算坡度/坡向中...]")
dz_dy, dz_dx = np.gradient(z, dy, dx)

slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg = np.degrees(slope_rad)

aspect_rad = np.arctan2(-dz_dx, dz_dy)
aspect_deg = np.degrees(aspect_rad)
aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

interp_slope = RegularGridInterpolator((y_coords, x_coords), slope_deg, bounds_error=False, fill_value=np.nan)
interp_aspect = RegularGridInterpolator((y_coords, x_coords), aspect_deg, bounds_error=False, fill_value=np.nan)


# ============================================================
# 7. 特征提取
# ============================================================
road["line_angle"] = compute_line_angles(road)

def extract_features(points_df, name="points", has_line_angle=False):
    print(f"\n开始提取 [{name}] 特征...")
    df = points_df.copy().reset_index(drop=True)
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)

    print(f"[{name}] 1/6 栅格特征插值中...")
    df["elevation"] = sample_interp(interp_elev, xs, ys)
    df["slope_deg"] = sample_interp(interp_slope, xs, ys)
    df["aspect_deg"] = sample_interp(interp_aspect, xs, ys)

    relief_vals, tpi_vals, rough_vals = local_window_features(
        z, x_coords, y_coords, xs, ys, desc=f"{name}-3x3局部窗口特征"
    )
    df["local_relief_3x3"] = relief_vals
    df["tpi_3x3"] = tpi_vals
    df["roughness_3x3"] = rough_vals

    print(f"[{name}] 2/6 线距离计算中...")
    df["dist_to_ridge1"] = query_nearest_segment_distance(
        ridge1_tree, xs, ys, desc=f"{name}-到一级分水岭"
    )
    df["dist_to_ridge2"] = query_nearest_segment_distance(
        ridge2_tree, xs, ys, desc=f"{name}-到二级分水岭"
    )
    df["dist_to_river"] = query_nearest_segment_distance(
        river_tree, xs, ys, desc=f"{name}-到河网"
    )

    if name == "秦直道全线":
        df["dist_to_road"] = 0.0
    else:
        df["dist_to_road"] = query_nearest_segment_distance(
            road_tree, xs, ys, desc=f"{name}-到秦直道"
        )

    print(f"[{name}] 3/6 点距离计算中...")
    df["dist_to_sites"] = nearest_point_distance(xs, ys, sites, desc=f"{name}-到遗迹点")
    df["dist_to_beacon"] = nearest_point_distance(xs, ys, beacon, desc=f"{name}-到烽火台")
    df["dist_to_pass"] = nearest_point_distance(xs, ys, pass_site, desc=f"{name}-到关隘")

    print(f"[{name}] 4/6 DEM 有效性检查中...")
    in_extent, in_valid = check_points_status_by_interp(xs, ys, xmin, xmax, ymin, ymax, interp_elev)
    df["in_dem_extent"] = in_extent
    df["in_dem_valid_mask"] = in_valid

    print(f"[{name}] 5/6 方向关系计算中...")
    nearest_r1_angle = query_nearest_segment_angle(
        ridge1_tree, ridge1_seg_angles, xs, ys, desc=f"{name}-最近一级分水岭方向"
    )

    if has_line_angle and "line_angle" in df.columns:
        df["angle_diff_to_ridge1"] = angle_diff_deg(
            df["line_angle"].to_numpy(dtype=float), nearest_r1_angle
        )
    else:
        df["angle_diff_to_ridge1"] = np.nan

    print(f"[{name}] 6/6 分水岭相对高程中...")
    pts = np.array([Point(x, y) for x, y in zip(xs, ys)], dtype=object)

    if len(pts) == 0:
        df["ridge1_nearest_elev"] = np.nan
        df["ridge1_relative_elev"] = np.nan
    else:
        idxs = ridge1_tree.query_nearest(pts, return_distance=False, all_matches=False)
        seg_idx = idxs[1]

        nearest_points = []
        for i, p in enumerate(pts):
            seg = ridge1_segs[seg_idx[i]]
            q = seg.interpolate(seg.project(p))
            nearest_points.append([q.x, q.y])

        nearest_points = np.asarray(nearest_points, dtype=float).reshape(-1, 2)

        ridge1_elev = sample_interp(
            interp_elev,
            nearest_points[:, 0],
            nearest_points[:, 1]
        )
        df["ridge1_nearest_elev"] = ridge1_elev
        df["ridge1_relative_elev"] = df["elevation"] - df["ridge1_nearest_elev"]

    print(f"[{name}] 提取完成，共 {len(df)} 个点")
    return df


# ============================================================
# 8. 道路特征（不再保留“只分析有效点”的旧思路）
# ============================================================
road_feat = extract_features(road, name="秦直道全线", has_line_angle=True)
road_feat = add_road_projection_info(road_feat, road_line)

print("\n[秦直道点数量]")
print("全线点数:", len(road_feat))
print("DEM 有效标记为 True 的数量:", int(road_feat["in_dem_valid_mask"].sum()))
print("DEM 有效占比:", road_feat["in_dem_valid_mask"].mean())

route_describe = summarize_describe(road_feat, RELATION_FEATURES, "road_all")


# ============================================================
# 9. 匹配背景点
# ============================================================
valid_rc = np.argwhere(~np.isnan(z))
print("\n[有效 DEM 格点数]", len(valid_rc))

np.random.seed(RANDOM_SEED)
bg_list = []
per_round = min(BG_PER_ROUND, len(valid_rc))

for rd in tqdm(range(BG_MAX_ROUNDS), desc="背景点候选抽样轮次"):
    sel = np.random.choice(len(valid_rc), size=per_round, replace=False)
    rc = valid_rc[sel]

    xs = x_coords[rc[:, 1]]
    ys = y_coords[rc[:, 0]]

    d_ridge1 = query_nearest_segment_distance(
        ridge1_tree, xs, ys, desc=f"第{rd+1}轮候选背景点-到一级分水岭距离"
    )
    d_road = query_nearest_segment_distance(
        road_tree, xs, ys, desc=f"第{rd+1}轮候选背景点-到秦直道距离"
    )

    keep = (
        (d_ridge1 <= RIDGE1_BG_MAX_DIST) &
        (d_road >= ROAD_BG_MIN_DIST) &
        (d_road <= ROAD_BG_MAX_DIST)
    )

    if np.any(keep):
        sub = pd.DataFrame({
            "x": xs[keep],
            "y": ys[keep],
            "dist_to_ridge1_precheck": d_ridge1[keep],
            "dist_to_road_precheck": d_road[keep],
        })
        bg_list.append(sub)

    curr_n = sum(len(t) for t in bg_list)
    tqdm.write(f"候选背景点抽样轮次 {rd+1}/{BG_MAX_ROUNDS}，当前候选数 = {curr_n}")
    if curr_n >= TARGET_BG_N * 3:
        break

if len(bg_list) == 0:
    raise RuntimeError("未抽到背景点，请放宽筛选参数。")

bg_candidates = pd.concat(bg_list, ignore_index=True).drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
print("\n[背景点候选池数量]", len(bg_candidates))

bg_candidates_feat = extract_features(bg_candidates[["x", "y"]], name="候选背景点", has_line_angle=False)
bg_feat = match_background_by_ridge_dist(
    road_df=road_feat,
    bg_cand=bg_candidates_feat,
    target_n=TARGET_BG_N,
    bins=MATCH_BINS,
    random_state=RANDOM_SEED
).reset_index(drop=True)

bg_feat = add_road_projection_info(bg_feat, road_line)
print("\n[最终匹配背景点数量]", len(bg_feat))

bg_describe = summarize_describe(bg_feat, ROUTE_COMPARE_FEATURES, "matched_background")
route_compare_stats = summarize_tests(
    road_feat, bg_feat, ROUTE_COMPARE_FEATURES,
    name_a="road", name_b="bg"
)


# ============================================================
# 10. 遗迹点特征（不再按“有效点”筛）
# ============================================================
beacon_feat = extract_features(beacon, name="烽火台", has_line_angle=False)
pass_feat = extract_features(pass_site, name="关隘", has_line_angle=False)
other_feat = extract_features(other_sites, name="其他遗存", has_line_angle=False)

beacon_feat = add_road_projection_info(beacon_feat, road_line)
pass_feat = add_road_projection_info(pass_feat, road_line)
other_feat = add_road_projection_info(other_feat, road_line)

beacon_describe = summarize_describe(beacon_feat, SITE_FEATURES, "beacon")
pass_describe = summarize_describe(pass_feat, SITE_FEATURES, "pass")
other_describe = summarize_describe(other_feat, SITE_FEATURES, "other_sites")

beacon_vs_road = summarize_tests(beacon_feat, road_feat, SITE_FEATURES, name_a="beacon", name_b="road")
pass_vs_road = summarize_tests(pass_feat, road_feat, SITE_FEATURES, name_a="pass", name_b="road")

if len(beacon_feat) > 0:
    beacon_chain = beacon_feat.sort_values("road_chainage_m").reset_index(drop=True).copy()
    beacon_chain["next_chainage_m"] = beacon_chain["road_chainage_m"].shift(-1)
    beacon_chain["相邻烽火台沿路线间距/m"] = beacon_chain["next_chainage_m"] - beacon_chain["road_chainage_m"]

    beacon_xy = beacon_chain[["x", "y"]].to_numpy(dtype=float)
    eu_dist = np.sqrt(np.sum(np.diff(beacon_xy, axis=0) ** 2, axis=1)) if len(beacon_xy) >= 2 else np.array([])
    beacon_chain["相邻烽火台欧氏距离/m"] = np.append(eu_dist, np.nan)

    beacon_spacing_summary = pd.DataFrame([{
        "烽火台数量": len(beacon_chain),
        "沿路线排序后相邻间距均值/m": beacon_chain["相邻烽火台沿路线间距/m"].dropna().mean(),
        "沿路线排序后相邻间距中位数/m": beacon_chain["相邻烽火台沿路线间距/m"].dropna().median(),
        "欧氏相邻间距均值/m": beacon_chain["相邻烽火台欧氏距离/m"].dropna().mean(),
        "欧氏相邻间距中位数/m": beacon_chain["相邻烽火台欧氏距离/m"].dropna().median(),
    }])
else:
    beacon_chain = pd.DataFrame()
    beacon_spacing_summary = pd.DataFrame()

pass_chain = pass_feat.sort_values("road_chainage_m").reset_index(drop=True).copy() if len(pass_feat) > 0 else pd.DataFrame()


# ============================================================
# 11. 特征关系分析
# ============================================================
road_relation_df = road_feat[RELATION_FEATURES].copy()

pearson_corr, spearman_corr = compute_correlations(road_relation_df, RELATION_FEATURES)
vif_table = compute_vif_table(road_relation_df, RELATION_FEATURES)
pca_loading, pca_explained, pca_scores = compute_pca(road_relation_df, RELATION_FEATURES, n_components=5)
ridge_stratified_summary = stratified_route_summary_by_ridge(road_feat, bins=5)

if SAVE_FIG:
    plot_corr_heatmap(spearman_corr, "道路点 Spearman 相关矩阵", "问题2_相关矩阵热力图_Spearman.png")
    print("已保存: 问题2_相关矩阵热力图_Spearman.png")

    plot_pca_scatter(pca_scores, "问题2_PCA前两主成分.png")
    print("已保存: 问题2_PCA前两主成分.png")


# ============================================================
# 12. 原则总结
# ============================================================
principle_summary = build_principle_summary(
    road_df=road_feat,
    bg_feat=bg_feat,
    beacon_feat=beacon_feat,
    pass_feat=pass_feat
)


# ============================================================
# 13. 可视化：烽火台沿路线分布
# ============================================================
if SAVE_FIG and len(beacon_feat) > 0:
    plt.figure(figsize=(10, 5.5))
    plt.scatter(beacon_feat["road_chainage_m"] / 1000.0, beacon_feat["elevation"], s=70)
    for i, row in beacon_feat.iterrows():
        plt.text(row["road_chainage_m"] / 1000.0, row["elevation"], f"{i+1}", fontsize=9)

    plt.xlabel("沿秦直道投影里程 / km")
    plt.ylabel("烽火台高程 / m")
    plt.title("烽火台沿路线分布")
    plt.tight_layout()
    plt.savefig("问题2_烽火台沿路线分布.png", dpi=FIG_DPI)
    plt.close()
    print("已保存: 问题2_烽火台沿路线分布.png")


# ============================================================
# 14. 导出 Excel
# ============================================================
with pd.ExcelWriter("result2.xlsx", engine="openpyxl") as writer:
    principle_summary.to_excel(writer, sheet_name="问题2原则总结", index=False)

    route_describe.to_excel(writer, sheet_name="道路点描述统计", index=False)
    bg_describe.to_excel(writer, sheet_name="匹配背景点描述统计", index=False)
    route_compare_stats.to_excel(writer, sheet_name="路线原则对比检验", index=False)

    road_feat.to_excel(writer, sheet_name="道路点特征", index=False)
    bg_feat.to_excel(writer, sheet_name="匹配背景点特征", index=False)

    beacon_describe.to_excel(writer, sheet_name="烽火台描述统计", index=False)
    pass_describe.to_excel(writer, sheet_name="关隘描述统计", index=False)
    other_describe.to_excel(writer, sheet_name="其他遗存描述统计", index=False)

    beacon_vs_road.to_excel(writer, sheet_name="烽火台_vs_道路", index=False)
    pass_vs_road.to_excel(writer, sheet_name="关隘_vs_道路", index=False)

    beacon_feat.to_excel(writer, sheet_name="烽火台特征明细", index=False)
    pass_feat.to_excel(writer, sheet_name="关隘特征明细", index=False)
    other_feat.to_excel(writer, sheet_name="其他遗存特征明细", index=False)

    if len(beacon_chain) > 0:
        beacon_chain.to_excel(writer, sheet_name="烽火台沿路线排序", index=False)
    if len(beacon_spacing_summary) > 0:
        beacon_spacing_summary.to_excel(writer, sheet_name="烽火台间距统计", index=False)
    if len(pass_chain) > 0:
        pass_chain.to_excel(writer, sheet_name="关隘沿路线排序", index=False)

    pearson_corr.to_excel(writer, sheet_name="Pearson相关矩阵")
    spearman_corr.to_excel(writer, sheet_name="Spearman相关矩阵")
    vif_table.to_excel(writer, sheet_name="VIF共线性分析", index=False)
    pca_loading.to_excel(writer, sheet_name="PCA载荷", index=False)
    pca_explained.to_excel(writer, sheet_name="PCA解释方差", index=False)
    pca_scores.to_excel(writer, sheet_name="PCA得分", index=False)
    ridge_stratified_summary.to_excel(writer, sheet_name="按分水岭距离分层统计", index=False)

print("\n已保存: result2.xlsx")
if SAVE_FIG:
    print("已保存: 问题2_总览图.png")
    if len(beacon_feat) > 0:
        print("已保存: 问题2_烽火台沿路线分布.png")
    print("已保存: 问题2_相关矩阵热力图_Spearman.png")
    print("已保存: 问题2_PCA前两主成分.png")

print("\n程序执行完成。")