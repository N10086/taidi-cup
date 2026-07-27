# -*- coding: utf-8 -*-
import os
import math
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from shapely.geometry import LineString, Point

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 参数区
# ============================================================
DEM_CSV = "陕甘八县的高程数据.csv"
ATT2_XLSX = "附件2  秦直道及周边地形和相关遗迹的数据.xlsx"
Q3_XLSX = "result3.xlsx"

OUT_XLSX = "result4.xlsx"
FIG1 = "问题4_设施规划总览图.png"
FIG2 = "问题4_设施链路示意图.png"

# 烽火台候选搜索半径（围绕新路线局部高点搜索）
BEACON_SEARCH_RADIUS = 800.0   # m
# 关隘沿路线局部搜索窗口（按路线里程搜索附近最优控制点）
PASS_LOCAL_SEARCH_RADIUS = 2500.0  # m

# 视通检测参数
LOS_SAMPLE_N = 80
LOS_OBSERVER_HEIGHT = 8.0  # m
LOS_TARGET_HEIGHT = 8.0    # m

# 距离约束
MIN_BEACON_SEPARATION = 500.0
MIN_PASS_SEPARATION = 12000.0

# 候选打分权重
# 烽火台：高势 + 贴分水岭 + 适度靠近路线 + 远离遗存
W_B_ELEV = 0.35
W_B_RIDGE = 0.25
W_B_ROUTE = 0.20
W_B_SLOPE = 0.10
W_B_RELATED = 0.10

# 关隘：控制性/瓶颈性 + 路线节点性
W_P_RELIEF = 0.25
W_P_ROUGH = 0.20
W_P_SLOPE = 0.20
W_P_RIVER = 0.15
W_P_RIDGE = 0.10
W_P_ROUTE = 0.10

assert os.path.exists(DEM_CSV), f"未找到文件: {DEM_CSV}"
assert os.path.exists(ATT2_XLSX), f"未找到文件: {ATT2_XLSX}"
assert os.path.exists(Q3_XLSX), f"未找到文件: {Q3_XLSX}"
print("文件检查通过。", flush=True)


# ============================================================
# 1. 日志与工具函数
# ============================================================
def log_step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def sample_interp(interp_obj, xs, ys):
    pts = np.column_stack([np.atleast_1d(ys), np.atleast_1d(xs)])
    return interp_obj(pts)


def polyline_length(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))


def cumulative_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.array([0.0])
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate([[0.0], np.cumsum(d)])


def nearest_grid_index(arr, vals):
    vals = np.asarray(vals)
    idx = np.searchsorted(arr, vals)
    idx = np.clip(idx, 1, len(arr) - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(arr[right] - vals) < np.abs(arr[left] - vals)
    return np.where(choose_right, right, left)


def q_of_valid(arr, q):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.quantile(a, q))


def minmax_norm_1d(arr):
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    mn = np.nanmin(arr[mask])
    mx = np.nanmax(arr[mask])
    if mx - mn < 1e-12:
        out[mask] = 0.0
    else:
        out[mask] = (arr[mask] - mn) / (mx - mn)
    return out


def classify_sites(sites_df):
    tp = sites_df["类型"].astype(str).str.strip()
    beacon = sites_df[tp.str.contains("烽火台", na=False)].copy()
    pass_site = sites_df[tp.str.contains("关隘", na=False)].copy()
    other_sites = sites_df[tp.str.contains("相关遗存", na=False)].copy()
    return beacon, pass_site, other_sites


def project_points_to_route_chainage(xs, ys, route_line):
    chainages = []
    near_xy = []

    for x, y in zip(xs, ys):
        p = Point(float(x), float(y))
        s = route_line.project(p)
        q = route_line.interpolate(s)
        chainages.append(s)
        near_xy.append([q.x, q.y])

    chainages = np.asarray(chainages, dtype=float)
    near_xy = np.asarray(near_xy, dtype=float).reshape(-1, 2)
    return chainages, near_xy


def point_at_chainage(route_line, s):
    q = route_line.interpolate(float(s))
    return np.array([q.x, q.y], dtype=float)


def extract_point_features(df_xy, interp_elev, interp_slope, interp_relief, interp_rough,
                           ridge1_tree, river_tree, route_tree, related_tree=None):
    xs = df_xy["x"].to_numpy(dtype=float)
    ys = df_xy["y"].to_numpy(dtype=float)
    pts = np.column_stack([xs, ys])

    out = df_xy.copy().reset_index(drop=True)
    out["elevation"] = sample_interp(interp_elev, xs, ys)
    out["slope_deg"] = sample_interp(interp_slope, xs, ys)
    out["local_relief_3x3"] = sample_interp(interp_relief, xs, ys)
    out["roughness_3x3"] = sample_interp(interp_rough, xs, ys)

    out["dist_to_ridge1"] = ridge1_tree.query(pts, k=1)[0]
    out["dist_to_river"] = river_tree.query(pts, k=1)[0]
    out["dist_to_route"] = route_tree.query(pts, k=1)[0]

    if related_tree is not None:
        out["dist_to_related"] = related_tree.query(pts, k=1)[0]
    else:
        out["dist_to_related"] = np.inf

    return out


def line_of_sight(p1, p2, interp_elev, n=80, h1=8.0, h2=8.0):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    ts = np.linspace(0, 1, n)
    xs = x1 + (x2 - x1) * ts
    ys = y1 + (y2 - y1) * ts
    elev = sample_interp(interp_elev, xs, ys)

    if np.any(~np.isfinite(elev)):
        return False

    z1 = elev[0] + h1
    z2 = elev[-1] + h2
    los_line = z1 + (z2 - z1) * ts

    terrain_mid = elev[1:-1]
    los_mid = los_line[1:-1]

    return np.all(terrain_mid <= los_mid)


def select_peaks_along_chainage(cand_df, score_col, min_spacing, top_k):
    if len(cand_df) == 0:
        return cand_df.copy()

    cand = cand_df.sort_values(score_col, ascending=False).copy().reset_index(drop=True)
    chosen = []

    for _, row in cand.iterrows():
        s = row["chainage_m"]
        ok = True
        for c in chosen:
            if abs(s - c["chainage_m"]) < min_spacing:
                ok = False
                break
        if ok:
            chosen.append(row.to_dict())
        if len(chosen) >= top_k:
            break

    if len(chosen) == 0:
        return cand_df.head(0).copy()

    return pd.DataFrame(chosen)


# ============================================================
# 2. 读取 DEM（修正版：强制翻转 z）
# ============================================================
log_step("开始读取 DEM...")
dem_raw = pd.read_csv(DEM_CSV, header=None)

log_step("开始解析 DEM...")
x_coords = pd.to_numeric(dem_raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
y_coords = pd.to_numeric(dem_raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)
z = dem_raw.iloc[1:, 1:].replace("NA", np.nan)
z = z.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

if not np.all(np.diff(x_coords) > 0):
    raise ValueError("x_coords 不是严格递增。")
if not np.all(np.diff(y_coords) > 0):
    raise ValueError("y_coords 不是严格递增，请检查 CSV。")

# 关键修正：只翻转 z
z = z[::-1, :]

dx = float(np.median(np.diff(x_coords)))
dy = float(np.median(np.diff(y_coords)))

interp_elev = RegularGridInterpolator(
    (y_coords, x_coords), z,
    method="linear",
    bounds_error=False,
    fill_value=np.nan
)

print("\n[DEM信息]")
print("shape:", z.shape)
print("x范围:", x_coords.min(), "~", x_coords.max())
print("y范围:", y_coords.min(), "~", y_coords.max())
print("高程范围:", np.nanmin(z), "~", np.nanmax(z), flush=True)


# ============================================================
# 3. 读取附件2
# ============================================================
log_step("开始读取附件2...")
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

beacon_raw, pass_raw, other_raw = classify_sites(sites)

print("\n[原始设施数量]")
print("原始烽火台数量:", len(beacon_raw))
print("原始关隘数量:", len(pass_raw))
print("原始相关遗存数量:", len(other_raw), flush=True)


# ============================================================
# 4. 读取第三问新路线
# ============================================================
log_step("开始读取第三问结果...")
new_route = pd.read_excel(Q3_XLSX, sheet_name="重规划路线坐标")
new_route = standardize_xy(new_route)

if len(new_route) < 2:
    raise RuntimeError("第三问新路线点数不足。")

new_route_line = LineString(new_route[["x", "y"]].to_numpy(dtype=float))
new_route_length = polyline_length(new_route["x"], new_route["y"])

print("\n[第三问新路线]")
print("点数:", len(new_route))
print("长度(m):", new_route_length, flush=True)


# ============================================================
# 5. 直接使用整条原秦直道做历史设施标定参考
# ============================================================
log_step("开始提取原秦直道高程并确认历史参考路线...")
road["elevation"] = sample_interp(
    interp_elev,
    road["x"].to_numpy(dtype=float),
    road["y"].to_numpy(dtype=float)
)
road["in_dem_valid_mask"] = np.isfinite(road["elevation"])

# 不再采用“最长连续有效段”旧思路
road_base = road.copy().reset_index(drop=True)
old_route_line = LineString(road_base[["x", "y"]].to_numpy(dtype=float))
old_route_length = polyline_length(road_base["x"], road_base["y"])

print("\n[原秦直道基础信息]")
print("全线点数:", len(road_base))
print("DEM 有效标记为 True 的数量:", int(road_base["in_dem_valid_mask"].sum()))
print("DEM 有效占比:", road_base["in_dem_valid_mask"].mean())
print("原秦直道长度(m):", old_route_length, flush=True)


# ============================================================
# 6. 计算坡度 / 局部起伏 / 粗糙度插值器
# ============================================================
log_step("开始计算坡度...")
dz_dy, dz_dx = np.gradient(z, dy, dx)
slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
interp_slope = RegularGridInterpolator((y_coords, x_coords), slope_deg, bounds_error=False, fill_value=np.nan)

log_step("开始计算局部起伏与粗糙度...")
h, w = z.shape
relief = np.full((h, w), np.nan, dtype=float)
rough = np.full((h, w), np.nan, dtype=float)

for i in tqdm(range(h), desc="全图局部起伏/粗糙度"):
    r0 = max(0, i - 1)
    r1 = min(h, i + 2)
    for j in range(w):
        c0 = max(0, j - 1)
        c1 = min(w, j + 2)
        win = z[r0:r1, c0:c1]
        vals = win[np.isfinite(win)]
        if len(vals) == 0:
            continue
        relief[i, j] = np.max(vals) - np.min(vals)
        rough[i, j] = np.std(vals)

interp_relief = RegularGridInterpolator((y_coords, x_coords), relief, bounds_error=False, fill_value=np.nan)
interp_rough = RegularGridInterpolator((y_coords, x_coords), rough, bounds_error=False, fill_value=np.nan)


# ============================================================
# 7. 建立树
# ============================================================
log_step("开始建立空间索引...")
ridge1_tree = cKDTree(ridge1[["x", "y"]].to_numpy(dtype=float))
river_tree = cKDTree(river[["x", "y"]].to_numpy(dtype=float))
new_route_tree = cKDTree(new_route[["x", "y"]].to_numpy(dtype=float))
old_route_tree = cKDTree(road_base[["x", "y"]].to_numpy(dtype=float))
related_tree = cKDTree(other_raw[["x", "y"]].to_numpy(dtype=float)) if len(other_raw) > 0 else None


# ============================================================
# 8. 用原始设施反推“设施网络”参数
# ============================================================
log_step("开始从原始设施反推网络参数...")

beacon_raw_feat = extract_point_features(
    beacon_raw[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, river_tree, old_route_tree, related_tree
)
beacon_raw_feat = beacon_raw_feat[np.isfinite(beacon_raw_feat["elevation"])].copy().reset_index(drop=True)

pass_raw_feat = extract_point_features(
    pass_raw[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, river_tree, old_route_tree, related_tree
)
pass_raw_feat = pass_raw_feat[np.isfinite(pass_raw_feat["elevation"])].copy().reset_index(drop=True)

# 投影到整条原秦直道
if len(beacon_raw_feat) > 0:
    beacon_chain_old, _ = project_points_to_route_chainage(
        beacon_raw_feat["x"].to_numpy(),
        beacon_raw_feat["y"].to_numpy(),
        old_route_line
    )
    beacon_raw_feat["chainage_m"] = beacon_chain_old
    beacon_raw_feat = beacon_raw_feat.sort_values("chainage_m").reset_index(drop=True)
    old_beacon_spacing = np.diff(beacon_raw_feat["chainage_m"].to_numpy())
    beacon_spacing_med = np.median(old_beacon_spacing) if len(old_beacon_spacing) > 0 else 3500.0
    beacon_spacing_q75 = np.quantile(old_beacon_spacing, 0.75) if len(old_beacon_spacing) > 0 else beacon_spacing_med * 1.3
else:
    beacon_spacing_med = 3500.0
    beacon_spacing_q75 = 4500.0

if len(pass_raw_feat) > 0:
    pass_chain_old, _ = project_points_to_route_chainage(
        pass_raw_feat["x"].to_numpy(),
        pass_raw_feat["y"].to_numpy(),
        old_route_line
    )
    pass_raw_feat["chainage_m"] = pass_chain_old
    pass_raw_feat = pass_raw_feat.sort_values("chainage_m").reset_index(drop=True)

orig_beacon_count = len(beacon_raw_feat)
orig_pass_count = len(pass_raw_feat)

length_ratio = new_route_length / old_route_length if old_route_length > 1e-12 else 1.0
target_pass_count = max(2, int(round(orig_pass_count * length_ratio))) if orig_pass_count > 0 else 3
target_beacon_count_ref = max(8, int(round(orig_beacon_count * length_ratio))) if orig_beacon_count > 0 else 35

max_relay_spacing = max(beacon_spacing_q75, beacon_spacing_med * 1.15)
min_pass_spacing = max(MIN_PASS_SEPARATION, new_route_length / (target_pass_count + 1) * 0.6)

param_df = pd.DataFrame({
    "参数名": [
        "orig_beacon_count", "orig_pass_count", "old_route_length",
        "new_route_length", "length_ratio",
        "beacon_spacing_median", "beacon_spacing_q75", "max_relay_spacing",
        "target_pass_count", "target_beacon_count_ref", "min_pass_spacing"
    ],
    "数值": [
        orig_beacon_count, orig_pass_count, old_route_length,
        new_route_length, length_ratio,
        beacon_spacing_med, beacon_spacing_q75, max_relay_spacing,
        target_pass_count, target_beacon_count_ref, min_pass_spacing
    ],
    "解释": [
        "原始烽火台数量（DEM有效）",
        "原始关隘数量（DEM有效）",
        "整条原秦直道长度",
        "第三问新路线长度",
        "新旧长度比例",
        "原始烽火台沿路线间距中位数",
        "原始烽火台沿路线间距75%分位数",
        "第四问中相邻通信节点允许的最大推荐间距",
        "按新旧路线比例缩放得到的目标关隘数",
        "按新旧路线比例缩放得到的目标烽火台参考数",
        "关隘沿路线最小间距"
    ]
})

print("\n[第四问数据驱动标定参数]")
print(param_df, flush=True)


# ============================================================
# 9. 先规划关隘
# ============================================================
log_step("开始规划关隘...")

new_route_feat = extract_point_features(
    new_route[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, river_tree, new_route_tree, related_tree
)
new_route_feat["chainage_m"] = cumulative_distance(new_route_feat["x"], new_route_feat["y"])

inv_river = 1.0 / (new_route_feat["dist_to_river"].to_numpy(dtype=float) + 1.0)

relief_n = minmax_norm_1d(new_route_feat["local_relief_3x3"])
rough_n = minmax_norm_1d(new_route_feat["roughness_3x3"])
slope_n = minmax_norm_1d(new_route_feat["slope_deg"])
inv_river_n = minmax_norm_1d(inv_river)
ridge_n = 1.0 - minmax_norm_1d(new_route_feat["dist_to_ridge1"])
route_n = 1.0 - minmax_norm_1d(new_route_feat["dist_to_route"])

pass_score = (
    W_P_RELIEF * relief_n +
    W_P_ROUGH  * rough_n +
    W_P_SLOPE  * slope_n +
    W_P_RIVER  * inv_river_n +
    W_P_RIDGE  * ridge_n +
    W_P_ROUTE  * route_n
)

new_route_feat["pass_score"] = pass_score

end_boost = np.zeros(len(new_route_feat), dtype=float)
route_len = new_route_feat["chainage_m"].iloc[-1]
if route_len > 0:
    dist_to_start = new_route_feat["chainage_m"].to_numpy()
    dist_to_end = route_len - dist_to_start
    end_boost[(dist_to_start < route_len * 0.10)] += 0.05
    end_boost[(dist_to_end < route_len * 0.10)] += 0.05
new_route_feat["pass_score"] += end_boost

pass_candidates = new_route_feat[[
    "x", "y", "chainage_m", "elevation", "slope_deg",
    "local_relief_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_river", "pass_score"
]].copy()

planned_passes = select_peaks_along_chainage(
    pass_candidates,
    score_col="pass_score",
    min_spacing=min_pass_spacing,
    top_k=target_pass_count
).sort_values("chainage_m").reset_index(drop=True)

planned_passes["设施类型"] = "关隘"
planned_passes["编号"] = [f"P{i+1}" for i in range(len(planned_passes))]

print("\n[规划关隘结果]")
print(planned_passes[["编号", "chainage_m", "x", "y", "pass_score"]], flush=True)


# ============================================================
# 10. 再规划烽火台
# ============================================================
log_step("开始规划烽火台...")

anchor_nodes = []

anchor_nodes.append({
    "node_type": "起点",
    "node_id": "START",
    "chainage_m": 0.0,
    "x": new_route.iloc[0]["x"],
    "y": new_route.iloc[0]["y"]
})

for _, row in planned_passes.iterrows():
    anchor_nodes.append({
        "node_type": "关隘",
        "node_id": row["编号"],
        "chainage_m": row["chainage_m"],
        "x": row["x"],
        "y": row["y"]
    })

anchor_nodes.append({
    "node_type": "终点",
    "node_id": "END",
    "chainage_m": route_len,
    "x": new_route.iloc[-1]["x"],
    "y": new_route.iloc[-1]["y"]
})

anchor_df = pd.DataFrame(anchor_nodes).sort_values("chainage_m").reset_index(drop=True)

route_line = new_route_line

def search_beacon_candidate_near_chainage(chainage_target, prev_node_xy=None):
    center_xy = point_at_chainage(route_line, chainage_target)
    cx, cy = center_xy[0], center_xy[1]

    ix = nearest_grid_index(x_coords, [cx])[0]
    iy = nearest_grid_index(y_coords, [cy])[0]

    rx = int(np.ceil(BEACON_SEARCH_RADIUS / dx))
    ry = int(np.ceil(BEACON_SEARCH_RADIUS / dy))

    r0 = max(0, iy - ry)
    r1 = min(len(y_coords), iy + ry + 1)
    c0 = max(0, ix - rx)
    c1 = min(len(x_coords), ix + rx + 1)

    xs_win = x_coords[c0:c1]
    ys_win = y_coords[r0:r1]
    zz_win = z[r0:r1, c0:c1]

    cand_rows = []
    for rr, yv in enumerate(ys_win):
        for cc, xv in enumerate(xs_win):
            zc = zz_win[rr, cc]
            if not np.isfinite(zc):
                continue

            d_to_center = math.hypot(xv - cx, yv - cy)
            if d_to_center > BEACON_SEARCH_RADIUS:
                continue

            elev = zc
            slope_val = sample_interp(interp_slope, [xv], [yv])[0]
            relief_val = sample_interp(interp_relief, [xv], [yv])[0]
            rough_val = sample_interp(interp_rough, [xv], [yv])[0]
            ridge_d = ridge1_tree.query([[xv, yv]], k=1)[0][0]
            route_d = new_route_tree.query([[xv, yv]], k=1)[0][0]

            if related_tree is not None:
                related_d = related_tree.query([[xv, yv]], k=1)[0][0]
            else:
                related_d = np.inf

            cand_rows.append({
                "x": xv,
                "y": yv,
                "elevation": elev,
                "slope_deg": slope_val,
                "local_relief_3x3": relief_val,
                "roughness_3x3": rough_val,
                "dist_to_ridge1": ridge_d,
                "dist_to_route": route_d,
                "dist_to_related": related_d,
                "dist_to_center": d_to_center
            })

    if len(cand_rows) == 0:
        return None

    cand_df = pd.DataFrame(cand_rows)

    elev_n = minmax_norm_1d(cand_df["elevation"])
    ridge_n = 1.0 - minmax_norm_1d(cand_df["dist_to_ridge1"])
    route_n = 1.0 - minmax_norm_1d(cand_df["dist_to_route"])
    slope_n = 1.0 - minmax_norm_1d(cand_df["slope_deg"])
    related_n = minmax_norm_1d(cand_df["dist_to_related"])

    cand_df["beacon_score"] = (
        W_B_ELEV * elev_n +
        W_B_RIDGE * ridge_n +
        W_B_ROUTE * route_n +
        W_B_SLOPE * slope_n +
        W_B_RELATED * related_n
    )

    cand_df = cand_df.sort_values("beacon_score", ascending=False).reset_index(drop=True)

    if prev_node_xy is not None:
        for _, row in cand_df.head(40).iterrows():
            ok = line_of_sight(
                prev_node_xy,
                np.array([row["x"], row["y"]], dtype=float),
                interp_elev,
                n=LOS_SAMPLE_N,
                h1=LOS_OBSERVER_HEIGHT,
                h2=LOS_TARGET_HEIGHT
            )
            if ok:
                return row.to_dict()

    return cand_df.iloc[0].to_dict()


planned_beacons = []
prev_node_xy = np.array([anchor_df.iloc[0]["x"], anchor_df.iloc[0]["y"]], dtype=float)

for idx in range(len(anchor_df) - 1):
    left = anchor_df.iloc[idx]
    right = anchor_df.iloc[idx + 1]

    seg_start = left["chainage_m"]
    seg_end = right["chainage_m"]
    seg_len = seg_end - seg_start

    if seg_len <= 0:
        continue

    needed_intervals = int(np.ceil(seg_len / max_relay_spacing))
    needed_beacons = max(0, needed_intervals - 1)

    if needed_beacons == 0:
        prev_node_xy = np.array([right["x"], right["y"]], dtype=float)
        continue

    target_chainages = np.linspace(seg_start, seg_end, needed_beacons + 2)[1:-1]

    for s in target_chainages:
        cand = search_beacon_candidate_near_chainage(s, prev_node_xy=prev_node_xy)
        if cand is None:
            continue

        too_close = False
        cand_xy = np.array([cand["x"], cand["y"]], dtype=float)

        for b in planned_beacons:
            if math.hypot(cand_xy[0] - b["x"], cand_xy[1] - b["y"]) < MIN_BEACON_SEPARATION:
                too_close = True
                break

        for _, prow in planned_passes.iterrows():
            if math.hypot(cand_xy[0] - prow["x"], cand_xy[1] - prow["y"]) < MIN_BEACON_SEPARATION:
                too_close = True
                break

        if too_close:
            continue

        cand["chainage_m"] = s
        planned_beacons.append(cand)
        prev_node_xy = cand_xy

    prev_node_xy = np.array([right["x"], right["y"]], dtype=float)

planned_beacons = pd.DataFrame(planned_beacons)
if len(planned_beacons) > 0:
    planned_beacons = planned_beacons.sort_values("chainage_m").reset_index(drop=True)
    planned_beacons["设施类型"] = "烽火台"
    planned_beacons["编号"] = [f"B{i+1}" for i in range(len(planned_beacons))]

print("\n[规划烽火台结果]")
if len(planned_beacons) > 0:
    print(planned_beacons[["编号", "chainage_m", "x", "y", "beacon_score"]], flush=True)
else:
    print("未生成烽火台，请检查参数。", flush=True)


# ============================================================
# 11. 生成设施网络链路表
# ============================================================
log_step("开始生成设施链路表...")

network_rows = []

network_rows.append({
    "节点顺序": 1,
    "节点类型": "起点",
    "节点编号": "START",
    "chainage_m": 0.0,
    "x": anchor_df.iloc[0]["x"],
    "y": anchor_df.iloc[0]["y"]
})

mid_nodes = []

for _, row in planned_passes.iterrows():
    mid_nodes.append({
        "节点类型": "关隘",
        "节点编号": row["编号"],
        "chainage_m": row["chainage_m"],
        "x": row["x"],
        "y": row["y"]
    })

for _, row in planned_beacons.iterrows():
    mid_nodes.append({
        "节点类型": "烽火台",
        "节点编号": row["编号"],
        "chainage_m": row["chainage_m"],
        "x": row["x"],
        "y": row["y"]
    })

mid_df = pd.DataFrame(mid_nodes).sort_values("chainage_m").reset_index(drop=True) if len(mid_nodes) > 0 else pd.DataFrame(columns=["节点类型","节点编号","chainage_m","x","y"])

for i, row in mid_df.iterrows():
    network_rows.append({
        "节点顺序": len(network_rows) + 1,
        "节点类型": row["节点类型"],
        "节点编号": row["节点编号"],
        "chainage_m": row["chainage_m"],
        "x": row["x"],
        "y": row["y"]
    })

network_rows.append({
    "节点顺序": len(network_rows) + 1,
    "节点类型": "终点",
    "节点编号": "END",
    "chainage_m": route_len,
    "x": anchor_df.iloc[-1]["x"],
    "y": anchor_df.iloc[-1]["y"]
})

network_df = pd.DataFrame(network_rows).sort_values("chainage_m").reset_index(drop=True)
network_df["节点顺序"] = np.arange(1, len(network_df) + 1)

link_rows = []
for i in range(len(network_df) - 1):
    a = network_df.iloc[i]
    b = network_df.iloc[i + 1]

    dist_chain = b["chainage_m"] - a["chainage_m"]
    dist_euclid = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    vis_ok = line_of_sight(
        np.array([a["x"], a["y"]], dtype=float),
        np.array([b["x"], b["y"]], dtype=float),
        interp_elev,
        n=LOS_SAMPLE_N,
        h1=LOS_OBSERVER_HEIGHT,
        h2=LOS_TARGET_HEIGHT
    )

    link_rows.append({
        "起点节点": a["节点编号"],
        "起点类型": a["节点类型"],
        "终点节点": b["节点编号"],
        "终点类型": b["节点类型"],
        "沿路线间距/m": dist_chain,
        "欧氏距离/m": dist_euclid,
        "视通性": "是" if vis_ok else "否"
    })

link_df = pd.DataFrame(link_rows)


# ============================================================
# 12. 输出汇总表
# ============================================================
summary_df = pd.DataFrame({
    "指标": [
        "原始烽火台数量",
        "原始关隘数量",
        "规划烽火台数量",
        "规划关隘数量",
        "原始秦直道长度(m)",
        "新路线长度(m)",
        "原始烽火台中位间距(m)",
        "原始烽火台75%分位间距(m)",
        "规划网络最大推荐传讯间距(m)"
    ],
    "数值": [
        orig_beacon_count,
        orig_pass_count,
        len(planned_beacons),
        len(planned_passes),
        old_route_length,
        new_route_length,
        beacon_spacing_med,
        beacon_spacing_q75,
        max_relay_spacing
    ]
})


# ============================================================
# 13. 可视化
# ============================================================
log_step("开始绘图...")

margin = 12000.0
xmin = min(new_route["x"].min(), road_base["x"].min()) - margin
xmax = max(new_route["x"].max(), road_base["x"].max()) + margin
ymin = min(new_route["y"].min(), road_base["y"].min()) - margin
ymax = max(new_route["y"].max(), road_base["y"].max()) + margin

x_mask = (x_coords >= xmin) & (x_coords <= xmax)
y_mask = (y_coords >= ymin) & (y_coords <= ymax)

x_sub = x_coords[x_mask]
y_sub = y_coords[y_mask]
z_sub = z[np.ix_(y_mask, x_mask)]
extent = [x_sub.min(), x_sub.max(), y_sub.min(), y_sub.max()]

# 图1
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(z_sub, extent=extent, origin="lower", aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Elevation (m)")

ax.plot(river["x"], river["y"], linestyle="None", marker=".", markersize=1.0, alpha=0.30, label="河网")
ax.plot(ridge1["x"], ridge1["y"], linewidth=0.8, alpha=0.60, label="一级分水岭")
ax.plot(new_route["x"], new_route["y"], linewidth=2.0, color="red", label="第三问新路线")

if len(planned_beacons) > 0:
    ax.scatter(planned_beacons["x"], planned_beacons["y"], s=55, color="orange", label="规划烽火台")
if len(planned_passes) > 0:
    ax.scatter(planned_passes["x"], planned_passes["y"], s=75, color="blue", marker="s", label="规划关隘")
if len(other_raw) > 0:
    ax.scatter(other_raw["x"], other_raw["y"], s=35, color="green", marker="x", label="相关遗存")

ax.set_title("第四问：新路线上的设施重规划结果")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG1, dpi=300, bbox_inches="tight")
plt.close()

# 图2
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(network_df["chainage_m"] / 1000.0, np.zeros(len(network_df)), color="lightgray", linewidth=2)

for _, row in network_df.iterrows():
    xk = row["chainage_m"] / 1000.0
    if row["节点类型"] == "起点":
        ax.scatter(xk, 0, s=90, color="black", marker="o")
    elif row["节点类型"] == "终点":
        ax.scatter(xk, 0, s=100, color="black", marker="^")
    elif row["节点类型"] == "关隘":
        ax.scatter(xk, 0, s=95, color="blue", marker="s")
    else:
        ax.scatter(xk, 0, s=65, color="orange", marker="o")

    ax.text(xk, 0.03, row["节点编号"], fontsize=8, ha="center")

ax.set_title("第四问：沿新路线的设施链路示意图")
ax.set_xlabel("沿新路线里程 / km")
ax.set_yticks([])
plt.tight_layout()
plt.savefig(FIG2, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 14. 保存结果
# ============================================================
log_step("开始保存结果...")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="结果汇总", index=False)
    param_df.to_excel(writer, sheet_name="参数标定", index=False)
    planned_passes.to_excel(writer, sheet_name="规划关隘", index=False)
    planned_beacons.to_excel(writer, sheet_name="规划烽火台", index=False)
    network_df.to_excel(writer, sheet_name="设施网络节点", index=False)
    link_df.to_excel(writer, sheet_name="设施链路关系", index=False)
    new_route.to_excel(writer, sheet_name="第三问新路线", index=False)

print("\n已保存:", OUT_XLSX)
print("已保存图片：")
print(FIG1)
print(FIG2)

print("\n[第四问结果汇总]")
print(summary_df, flush=True)

print("\n说明：")
print("1) 关隘先作为战略控制节点进行规划；")
print("2) 烽火台不是独立选点，而是围绕‘起点-关隘-终点’形成连续传讯链；")
print("3) 相邻设施节点会检查视通性，因此设施之间具有联动约束；")
print("4) 输出的‘设施链路关系’表可直接用于论文说明设施间的相互呼应关系。")