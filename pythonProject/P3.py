# -*- coding: utf-8 -*-
import os
import heapq
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 参数区
# ============================================================
DEM_CSV = "陕甘八县的高程数据.csv"
ATT2_XLSX = "附件2  秦直道及周边地形和相关遗迹的数据.xlsx"

# -------------------------
# 数据驱动规划参数
# -------------------------
coarsen_factor = 4
bbox_margin = 10000.0

ridge1_q_soft1 = 0.25
ridge1_q_soft2 = 0.75

river_q_near1 = 0.25
river_q_near2 = 0.50

slope_q_hard = 0.975

# 历史交通走廊软阈值下限
road_soft_1_min = 800.0
road_soft_2_min = 2500.0

# 相关遗存避让
use_related_avoid = True
related_q_hard = 300.0
related_q_soft = 1200.0
w_related = 0.08

# 机制权重
W_GROUP_RIDGE = 0.35
W_GROUP_PASS  = 0.30
W_GROUP_RIVER = 0.15
W_GROUP_ROAD  = 0.15
W_GROUP_OTHER = 0.05

OUT_XLSX = "result3.xlsx"
FIG1 = "问题3_规划路线总览图.png"
FIG2 = "问题3_代价面与新路线.png"
FIG3 = "问题3_新旧路线指标对比.png"
FIG4 = "问题3_新旧路线剖面对比.png"
FIG5 = "问题3_局部偏移较大路段.png"

assert os.path.exists(DEM_CSV), f"未找到文件: {DEM_CSV}"
assert os.path.exists(ATT2_XLSX), f"未找到文件: {ATT2_XLSX}"
print("文件检查通过。", flush=True)


# ============================================================
# 1. 工具函数
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


def sample_interp(interp_obj, xs, ys):
    pts = np.column_stack([np.atleast_1d(ys), np.atleast_1d(xs)])
    return interp_obj(pts)


def compute_relief_roughness_with_progress(z, win_radius=1, desc="计算局部起伏与粗糙度"):
    h, w = z.shape
    relief = np.full((h, w), np.nan, dtype=float)
    rough = np.full((h, w), np.nan, dtype=float)

    for i in tqdm(range(h), desc=desc):
        r0 = max(0, i - win_radius)
        r1 = min(h, i + win_radius + 1)

        for j in range(w):
            c0 = max(0, j - win_radius)
            c1 = min(w, j + win_radius + 1)

            win = z[r0:r1, c0:c1]
            vals = win[np.isfinite(win)]
            if len(vals) == 0:
                continue

            relief[i, j] = np.max(vals) - np.min(vals)
            rough[i, j] = np.std(vals)

    return relief, rough


def block_reduce_mean(arr, factor):
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    arr2 = arr[:h2, :w2]
    arr_reshaped = arr2.reshape(h2 // factor, factor, w2 // factor, factor)
    return np.nanmean(arr_reshaped, axis=(1, 3))


def minmax_norm(arr, mask):
    out = np.full_like(arr, np.nan, dtype=float)
    vals = arr[mask & np.isfinite(arr)]
    if len(vals) == 0:
        return out
    mn = np.nanmin(vals)
    mx = np.nanmax(vals)
    if mx - mn < 1e-12:
        out[mask & np.isfinite(arr)] = 0.0
        return out
    out[mask & np.isfinite(arr)] = (arr[mask & np.isfinite(arr)] - mn) / (mx - mn)
    return out


def avoidance_penalty_array(dist_array, hard_limit, soft_limit):
    out = np.zeros_like(dist_array, dtype=float)
    out[dist_array < hard_limit] = np.inf
    mask = (dist_array >= hard_limit) & (dist_array < soft_limit)
    out[mask] = (soft_limit - dist_array[mask]) / (soft_limit - hard_limit)
    out[dist_array >= soft_limit] = 0.0
    return out


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


def q_of_valid(arr, q):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.quantile(a, q))


def query_tree_grid_distance(tree, grid_pts, out_shape, chunk_size=50000, desc="计算距离场"):
    out = np.empty(len(grid_pts), dtype=float)
    n_chunks = (len(grid_pts) + chunk_size - 1) // chunk_size

    for start in tqdm(range(0, len(grid_pts), chunk_size), total=n_chunks, desc=desc):
        end = min(start + chunk_size, len(grid_pts))
        out[start:end] = tree.query(grid_pts[start:end], k=1)[0]

    return out.reshape(out_shape)


def nearest_valid_cell(xy, xgrid, ygrid, mask, cost_surface):
    yy_idx, xx_idx = np.where(mask & np.isfinite(cost_surface))
    valid_pts = np.column_stack([xgrid[xx_idx], ygrid[yy_idx]])
    if len(valid_pts) == 0:
        raise RuntimeError("没有有效候选网格点。")
    tree = cKDTree(valid_pts)
    _, idx = tree.query([xy], k=1)
    px, py = valid_pts[idx[0]]
    j = int(np.argmin(np.abs(xgrid - px)))
    i = int(np.argmin(np.abs(ygrid - py)))
    return (i, j), (float(px), float(py))


def heuristic(a, b, dx, dy):
    return np.sqrt(((a[0]-b[0]) * dy)**2 + ((a[1]-b[1]) * dx)**2)


def astar(cost, start, goal, dx, dy, progress_every=5000):
    H, W = cost.shape
    moves = [
        (-1, 0, dy), (1, 0, dy),
        (0, -1, dx), (0, 1, dx),
        (-1, -1, np.sqrt(dx**2 + dy**2)),
        (-1,  1, np.sqrt(dx**2 + dy**2)),
        ( 1, -1, np.sqrt(dx**2 + dy**2)),
        ( 1,  1, np.sqrt(dx**2 + dy**2)),
    ]

    open_heap = [(0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    visited = set()

    iter_count = 0
    pbar = tqdm(desc="A*搜索", unit="node")

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        iter_count += 1
        pbar.update(1)

        if iter_count % progress_every == 0:
            pbar.set_postfix({"visited": len(visited), "open": len(open_heap)})

        if current == goal:
            pbar.close()
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        ci, cj = current
        for di, dj, step_len in moves:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= H or nj < 0 or nj >= W:
                continue
            if not np.isfinite(cost[ni, nj]):
                continue

            step_cost = (cost[ci, cj] + cost[ni, nj]) / 2.0
            tentative_g = g_score[current] + step_cost * (1.0 + 0.15 * step_len / min(dx, dy))
            neighbor = (ni, nj)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + 0.3 * heuristic(neighbor, goal, dx, dy) / max(dx, dy)
                came_from[neighbor] = current
                heapq.heappush(open_heap, (f, neighbor))

    pbar.close()
    return None


def extract_route_features(route_df, interp_elev, interp_slope, interp_relief, interp_rough,
                           ridge1_tree, ridge2_tree, river_tree, road_tree, related_tree=None):
    xs = route_df["x"].to_numpy(dtype=float)
    ys = route_df["y"].to_numpy(dtype=float)
    pts = np.column_stack([xs, ys])

    feat = pd.DataFrame({
        "x": xs,
        "y": ys,
        "elevation": sample_interp(interp_elev, xs, ys),
        "slope_deg": sample_interp(interp_slope, xs, ys),
        "local_relief_3x3": sample_interp(interp_relief, xs, ys),
        "roughness_3x3": sample_interp(interp_rough, xs, ys),
        "cumdist": cumulative_distance(xs, ys)
    })

    feat["dist_to_ridge1"] = ridge1_tree.query(pts, k=1)[0]
    feat["dist_to_ridge2"] = ridge2_tree.query(pts, k=1)[0]
    feat["dist_to_river"] = river_tree.query(pts, k=1)[0]
    feat["dist_to_old_road"] = road_tree.query(pts, k=1)[0]

    if related_tree is not None:
        feat["dist_to_related"] = related_tree.query(pts, k=1)[0]
    else:
        feat["dist_to_related"] = np.inf

    return feat


def is_connected(mask, start_rc, end_rc):
    H, W = mask.shape
    sr, sc = start_rc
    gr, gc = end_rc

    if not mask[sr, sc] or not mask[gr, gc]:
        return False

    q = deque()
    q.append((sr, sc))
    visited = set([(sr, sc)])
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True

        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if not mask[nr, nc]:
                continue
            if (nr, nc) in visited:
                continue
            visited.add((nr, nc))
            q.append((nr, nc))

    return False


# ============================================================
# 2. 读取 DEM（修正版：强制翻转 z）
# ============================================================
log_step("开始读取 DEM...")
dem_raw = pd.read_csv(DEM_CSV, header=None)

log_step("开始解析 DEM 坐标与高程矩阵...")
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

related_sites = sites[sites["类型"].astype(str).str.contains("相关遗存", na=False)].copy().reset_index(drop=True)


# ============================================================
# 4. 计算全图坡度
# ============================================================
log_step("开始计算全图坡度...")
dz_dy, dz_dx = np.gradient(z, dy, dx)
slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
interp_slope = RegularGridInterpolator((y_coords, x_coords), slope_deg, bounds_error=False, fill_value=np.nan)


# ============================================================
# 5. 直接以整条原路线作为旧路线参考
# ============================================================
log_step("开始提取原路线高程并确认路线基准...")
road["elevation"] = sample_interp(
    interp_elev,
    road["x"].to_numpy(dtype=float),
    road["y"].to_numpy(dtype=float)
)
road["in_dem_valid_mask"] = np.isfinite(road["elevation"])

# 不再采用“最长连续有效段”旧思想
road_base = road.copy().reset_index(drop=True)

print("\n[原路线基础信息]")
print("全线点数:", len(road))
print("DEM 有效标记为 True 的数量:", int(road["in_dem_valid_mask"].sum()))
print("DEM 有效占比:", road["in_dem_valid_mask"].mean(), flush=True)


# ============================================================
# 6. 建立空间索引
# ============================================================
log_step("开始建立空间索引...")
ridge1_tree = cKDTree(ridge1[["x", "y"]].to_numpy(dtype=float))
ridge2_tree = cKDTree(ridge2[["x", "y"]].to_numpy(dtype=float))
river_tree = cKDTree(river[["x", "y"]].to_numpy(dtype=float))
road_tree = cKDTree(road_base[["x", "y"]].to_numpy(dtype=float))
related_tree = cKDTree(related_sites[["x", "y"]].to_numpy(dtype=float)) if len(related_sites) > 0 else None


# ============================================================
# 7. 裁剪研究区
# ============================================================
log_step("开始裁剪研究区...")
road_xmin = road_base["x"].min() - bbox_margin
road_xmax = road_base["x"].max() + bbox_margin
road_ymin = road_base["y"].min() - bbox_margin
road_ymax = road_base["y"].max() + bbox_margin

x_mask = (x_coords >= road_xmin) & (x_coords <= road_xmax)
y_mask = (y_coords >= road_ymin) & (y_coords <= road_ymax)

x_sub = x_coords[x_mask]
y_sub = y_coords[y_mask]
z_sub = z[np.ix_(y_mask, x_mask)]
slope_sub = slope_deg[np.ix_(y_mask, x_mask)]

print("\n[裁剪后区域]")
print("shape:", z_sub.shape, flush=True)


# ============================================================
# 8. 裁剪区局部起伏与粗糙度
# ============================================================
log_step("开始对裁剪区计算局部起伏度与粗糙度...")
relief_sub, rough_sub = compute_relief_roughness_with_progress(
    z_sub, win_radius=1, desc="裁剪区局部起伏/粗糙度"
)

interp_relief = RegularGridInterpolator((y_sub, x_sub), relief_sub, bounds_error=False, fill_value=np.nan)
interp_rough = RegularGridInterpolator((y_sub, x_sub), rough_sub, bounds_error=False, fill_value=np.nan)


# ============================================================
# 9. 提取原路线特征并标定参数
# ============================================================
log_step("开始提取原路线特征并标定参数...")
road_feat = extract_route_features(
    road_base[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, ridge2_tree, river_tree, road_tree, related_tree
)

ridge1_soft_1 = q_of_valid(road_feat["dist_to_ridge1"], ridge1_q_soft1)
ridge1_soft_2 = q_of_valid(road_feat["dist_to_ridge1"], ridge1_q_soft2)

river_near_1 = q_of_valid(road_feat["dist_to_river"], river_q_near1)
river_near_2 = q_of_valid(road_feat["dist_to_river"], river_q_near2)

slope_hard_limit = q_of_valid(road_feat["slope_deg"], slope_q_hard)

road_soft_1 = road_soft_1_min
road_soft_2 = road_soft_2_min

print("\n[第三问数据驱动参数标定]")
print(f"ridge1_soft_1 = {ridge1_soft_1:.2f}")
print(f"ridge1_soft_2 = {ridge1_soft_2:.2f}")
print(f"river_near_1 = {river_near_1:.2f}")
print(f"river_near_2 = {river_near_2:.2f}")
print(f"slope_hard_limit = {slope_hard_limit:.2f}")
print(f"road_soft_1 = {road_soft_1:.2f}")
print(f"road_soft_2 = {road_soft_2:.2f}", flush=True)


# ============================================================
# 10. 降采样
# ============================================================
log_step("开始降采样...")
z_c = block_reduce_mean(z_sub, coarsen_factor)
slope_c = block_reduce_mean(slope_sub, coarsen_factor)
relief_c = block_reduce_mean(relief_sub, coarsen_factor)
rough_c = block_reduce_mean(rough_sub, coarsen_factor)

x_sub2 = x_sub[: (len(x_sub) // coarsen_factor) * coarsen_factor]
y_sub2 = y_sub[: (len(y_sub) // coarsen_factor) * coarsen_factor]

x_c = x_sub2.reshape(-1, coarsen_factor).mean(axis=1)
y_c = y_sub2.reshape(-1, coarsen_factor).mean(axis=1)

dx_c = float(np.median(np.diff(x_c)))
dy_c = float(np.median(np.diff(y_c)))

print("\n[降采样后]")
print("shape:", z_c.shape)
print("dx_c =", dx_c, "dy_c =", dy_c, flush=True)


# ============================================================
# 11. 计算粗网格距离场
# ============================================================
log_step("开始计算粗网格距离场...")
XX, YY = np.meshgrid(x_c, y_c)
grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

dist_ridge1 = query_tree_grid_distance(ridge1_tree, grid_pts, z_c.shape, desc="到一级分水岭距离场")
dist_ridge2 = query_tree_grid_distance(ridge2_tree, grid_pts, z_c.shape, desc="到二级分水岭距离场")
dist_river = query_tree_grid_distance(river_tree, grid_pts, z_c.shape, desc="到河网距离场")
dist_road = query_tree_grid_distance(road_tree, grid_pts, z_c.shape, desc="到原路线距离场")

if related_tree is not None:
    dist_related = query_tree_grid_distance(related_tree, grid_pts, z_c.shape, desc="到相关遗存距离场")
else:
    dist_related = np.full_like(z_c, np.inf, dtype=float)


# ============================================================
# 12. 候选搜索区
# ============================================================
log_step("开始构建候选搜索区...")
valid_mask = np.isfinite(z_c)

# 直接用原路线首尾点
start_xy = road_base.iloc[0][["x", "y"]].to_numpy(dtype=float)
end_xy = road_base.iloc[-1][["x", "y"]].to_numpy(dtype=float)

base_mask = valid_mask.copy()
start_rc, start_snap = nearest_valid_cell(start_xy, x_c, y_c, base_mask, np.where(base_mask, 1.0, np.nan))
end_rc, end_snap = nearest_valid_cell(end_xy, x_c, y_c, base_mask, np.where(base_mask, 1.0, np.nan))

relax_plan = [
    {
        "name": "初始硬约束",
        "slope_hard_limit": slope_hard_limit,
        "use_related_avoid": use_related_avoid,
    },
    {
        "name": "放宽坡度",
        "slope_hard_limit": max(slope_hard_limit, 45.0),
        "use_related_avoid": use_related_avoid,
    },
    {
        "name": "再放宽坡度",
        "slope_hard_limit": max(slope_hard_limit, 55.0),
        "use_related_avoid": use_related_avoid,
    },
    {
        "name": "关闭遗存硬避让",
        "slope_hard_limit": max(slope_hard_limit, 55.0),
        "use_related_avoid": False,
    },
]

candidate_mask = None
chosen_plan = None

for plan in relax_plan:
    trial_mask = valid_mask & (slope_c <= plan["slope_hard_limit"])

    if plan["use_related_avoid"] and (related_tree is not None):
        trial_mask = trial_mask & (dist_related >= related_q_hard)

    connected = is_connected(trial_mask, start_rc, end_rc)

    print(
        f"方案={plan['name']} | 候选格点数={int(trial_mask.sum())} | "
        f"slope_hard_limit={plan['slope_hard_limit']:.2f} | "
        f"use_related_avoid={plan['use_related_avoid']} | "
        f"起终点连通={connected}",
        flush=True
    )

    if trial_mask.sum() > 0 and connected:
        candidate_mask = trial_mask
        chosen_plan = plan
        break

if candidate_mask is None:
    raise RuntimeError("即使只保留 DEM 有效区 + 坡度约束，起终点仍不连通，请检查裁剪区或端点选取。")

print("\n[最终采用候选区方案]")
print(chosen_plan, flush=True)

slope_hard_limit = chosen_plan["slope_hard_limit"]
use_related_avoid = chosen_plan["use_related_avoid"]


# ============================================================
# 13. 综合代价面
# ============================================================
log_step("开始构建综合代价面...")
ridge1_cost = np.full_like(dist_ridge1, np.nan, dtype=float)
ridge1_cost[(candidate_mask) & (dist_ridge1 <= ridge1_soft_1)] = 0.0
ridge1_cost[(candidate_mask) & (dist_ridge1 > ridge1_soft_1) & (dist_ridge1 <= ridge1_soft_2)] = 0.25
ridge1_cost[(candidate_mask) & (dist_ridge1 > ridge1_soft_2)] = 1.0

road_cost = np.full_like(dist_road, np.nan, dtype=float)
road_cost[(candidate_mask) & (dist_road <= road_soft_1)] = 0.0
road_cost[(candidate_mask) & (dist_road > road_soft_1) & (dist_road <= road_soft_2)] = 0.35
road_cost[(candidate_mask) & (dist_road > road_soft_2)] = 1.0

river_cost = np.full_like(dist_river, np.nan, dtype=float)
river_cost[(candidate_mask) & (dist_river <= river_near_1)] = 1.0
river_cost[(candidate_mask) & (dist_river > river_near_1) & (dist_river <= river_near_2)] = 0.45
river_cost[(candidate_mask) & (dist_river > river_near_2)] = 0.05

ridge2_n = minmax_norm(dist_ridge2, candidate_mask)
slope_n = minmax_norm(slope_c, candidate_mask)
relief_n = minmax_norm(relief_c, candidate_mask)
rough_n = minmax_norm(rough_c, candidate_mask)
elev_n = minmax_norm(z_c, candidate_mask)
elev_cost = 1 - elev_n

cost_group_ridge = ridge1_cost
cost_group_pass = np.where(candidate_mask, (0.45 * slope_n + 0.35 * relief_n + 0.20 * rough_n), np.nan)
cost_group_river = river_cost
cost_group_road = road_cost
cost_group_other = np.where(candidate_mask, (0.60 * ridge2_n + 0.40 * elev_cost), np.nan)

cost_surface = np.full_like(z_c, np.nan, dtype=float)
cost_surface[candidate_mask] = (
    W_GROUP_RIDGE * cost_group_ridge[candidate_mask] +
    W_GROUP_PASS  * cost_group_pass[candidate_mask] +
    W_GROUP_RIVER * cost_group_river[candidate_mask] +
    W_GROUP_ROAD  * cost_group_road[candidate_mask] +
    W_GROUP_OTHER * cost_group_other[candidate_mask]
)

cost_surface[(candidate_mask) & (dist_ridge1 > ridge1_soft_2)] += 0.8
cost_surface[(candidate_mask) & (dist_road > road_soft_2)] += 0.6

if use_related_avoid and related_tree is not None:
    related_penalty = avoidance_penalty_array(dist_related, related_q_hard, related_q_soft)
    cost_surface[(candidate_mask) & np.isinf(related_penalty)] = np.nan
    mask_soft = candidate_mask & (~np.isinf(related_penalty))
    cost_surface[mask_soft] += w_related * related_penalty[mask_soft]


# ============================================================
# 14. 输出起终点映射
# ============================================================
log_step("开始输出规划起终点映射...")
print("\n[规划起终点映射]")
print("原起点:", tuple(start_xy), "-> 粗格点:", start_rc, "坐标:", start_snap)
print("原终点:", tuple(end_xy), "-> 粗格点:", end_rc, "坐标:", end_snap, flush=True)


# ============================================================
# 15. A* 搜索
# ============================================================
log_step("开始 A* 路径搜索...")
path_rc = astar(cost_surface, start_rc, end_rc, dx_c, dy_c, progress_every=3000)
if path_rc is None:
    raise RuntimeError("A* 在连通候选区中仍未找到路径，请进一步检查代价面。")
print("A* 搜索完成，路径格点数:", len(path_rc), flush=True)


# ============================================================
# 16. 路径转坐标
# ============================================================
path_rc = np.asarray(path_rc, dtype=int)
new_route = pd.DataFrame({
    "序号": np.arange(1, len(path_rc) + 1),
    "x": x_c[path_rc[:, 1]],
    "y": y_c[path_rc[:, 0]]
})


# ============================================================
# 17. 提取新旧路线特征
# ============================================================
log_step("开始提取新旧路线特征...")
old_feat = extract_route_features(
    road_base[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, ridge2_tree, river_tree, road_tree, related_tree
)
new_feat = extract_route_features(
    new_route[["x", "y"]].copy(),
    interp_elev, interp_slope, interp_relief, interp_rough,
    ridge1_tree, ridge2_tree, river_tree, road_tree, related_tree
)


# ============================================================
# 18. 新旧路线对比
# ============================================================
compare_df = pd.DataFrame({
    "指标": [
        "路线长度(m)",
        "平均高程(m)",
        "平均坡度(°)",
        "平均局部起伏度",
        "平均粗糙度",
        "平均距一级分水岭(m)",
        "平均距二级分水岭(m)",
        "平均距河网(m)"
    ],
    "原秦直道": [
        polyline_length(old_feat["x"], old_feat["y"]),
        np.nanmean(old_feat["elevation"]),
        np.nanmean(old_feat["slope_deg"]),
        np.nanmean(old_feat["local_relief_3x3"]),
        np.nanmean(old_feat["roughness_3x3"]),
        np.nanmean(old_feat["dist_to_ridge1"]),
        np.nanmean(old_feat["dist_to_ridge2"]),
        np.nanmean(old_feat["dist_to_river"])
    ],
    "重规划路线": [
        polyline_length(new_feat["x"], new_feat["y"]),
        np.nanmean(new_feat["elevation"]),
        np.nanmean(new_feat["slope_deg"]),
        np.nanmean(new_feat["local_relief_3x3"]),
        np.nanmean(new_feat["roughness_3x3"]),
        np.nanmean(new_feat["dist_to_ridge1"]),
        np.nanmean(new_feat["dist_to_ridge2"]),
        np.nanmean(new_feat["dist_to_river"])
    ]
})
compare_df["绝对变化(新-旧)"] = compare_df["重规划路线"] - compare_df["原秦直道"]

mobility_metrics = ["路线长度(m)", "平均坡度(°)", "平均局部起伏度", "平均粗糙度"]
mobility_rows = []
for metric in mobility_metrics:
    old_val = compare_df.loc[compare_df["指标"] == metric, "原秦直道"].values[0]
    new_val = compare_df.loc[compare_df["指标"] == metric, "重规划路线"].values[0]
    mobility_rows.append({
        "指标": metric,
        "原秦直道": old_val,
        "重规划路线": new_val,
        "绝对改善量": old_val - new_val,
        "改善率(%)": (old_val - new_val) / old_val * 100 if abs(old_val) > 1e-12 else np.nan
    })
mobility_df = pd.DataFrame(mobility_rows)

spatial_shift_df = pd.DataFrame({
    "指标": [
        "平均高程变化(m)",
        "平均距一级分水岭变化(m)",
        "平均距二级分水岭变化(m)",
        "平均距河网变化(m)",
        "平均距相关遗存变化(m)"
    ],
    "变化量(新-旧)": [
        np.nanmean(new_feat["elevation"]) - np.nanmean(old_feat["elevation"]),
        np.nanmean(new_feat["dist_to_ridge1"]) - np.nanmean(old_feat["dist_to_ridge1"]),
        np.nanmean(new_feat["dist_to_ridge2"]) - np.nanmean(old_feat["dist_to_ridge2"]),
        np.nanmean(new_feat["dist_to_river"]) - np.nanmean(old_feat["dist_to_river"]),
        np.nanmean(new_feat["dist_to_related"]) - np.nanmean(old_feat["dist_to_related"])
    ]
})

param_df = pd.DataFrame({
    "参数名": [
        "coarsen_factor", "bbox_margin",
        "ridge1_soft_1", "ridge1_soft_2",
        "river_near_1", "river_near_2",
        "slope_hard_limit",
        "road_soft_1", "road_soft_2",
        "related_q_hard", "related_q_soft",
        "W_GROUP_RIDGE", "W_GROUP_PASS", "W_GROUP_RIVER", "W_GROUP_ROAD", "W_GROUP_OTHER",
        "chosen_candidate_plan"
    ],
    "数值": [
        coarsen_factor, bbox_margin,
        ridge1_soft_1, ridge1_soft_2,
        river_near_1, river_near_2,
        slope_hard_limit,
        road_soft_1, road_soft_2,
        related_q_hard, related_q_soft,
        W_GROUP_RIDGE, W_GROUP_PASS, W_GROUP_RIVER, W_GROUP_ROAD, W_GROUP_OTHER,
        chosen_plan["name"]
    ],
    "来源解释": [
        "计算效率需求",
        "整条原路线范围外扩",
        "原路线到一级分水岭距离25%分位数",
        "原路线到一级分水岭距离75%分位数",
        "原路线到河网距离25%分位数",
        "原路线到河网距离50%分位数",
        "经自动放宽后的最终坡度硬限制",
        "历史交通主走廊软阈值下限",
        "历史交通主走廊强惩罚阈值下限",
        "相关遗存硬避让半径",
        "相关遗存软避让半径",
        "分水岭依附机制权重",
        "通行性机制权重",
        "水文干扰机制权重",
        "历史走廊一致性权重",
        "次级补充机制权重",
        "候选区自动放宽后采用的方案名"
    ]
})

print("\n[新旧路线总对比]")
print(compare_df)
print("\n[通行性优化表]")
print(mobility_df)
print("\n[空间偏移表]")
print(spatial_shift_df, flush=True)


# ============================================================
# 19. 局部偏移分析
# ============================================================
new_pts = new_feat[["x", "y"]].to_numpy(dtype=float)
offset_to_old = road_tree.query(new_pts, k=1)[0]
new_feat["offset_to_old"] = offset_to_old

offset_threshold = np.percentile(offset_to_old, 85) if len(offset_to_old) > 0 else np.nan
new_feat["is_large_offset"] = new_feat["offset_to_old"] >= offset_threshold if np.isfinite(offset_threshold) else False

segments = []
flag = False
start_idx = None
for i, v in enumerate(new_feat["is_large_offset"].to_numpy()):
    if v and not flag:
        start_idx = i
        flag = True
    elif (not v) and flag:
        end_idx = i - 1
        segments.append((start_idx, end_idx))
        flag = False
if flag:
    segments.append((start_idx, len(new_feat) - 1))

segment_rows = []
for k, (s, e) in enumerate(segments, 1):
    seg = new_feat.iloc[s:e+1]
    segment_rows.append({
        "路段编号": k,
        "起点序号": s + 1,
        "终点序号": e + 1,
        "起点x": seg["x"].iloc[0],
        "起点y": seg["y"].iloc[0],
        "终点x": seg["x"].iloc[-1],
        "终点y": seg["y"].iloc[-1],
        "平均偏移距离(m)": seg["offset_to_old"].mean(),
        "平均高程(m)": seg["elevation"].mean(),
        "平均坡度(°)": seg["slope_deg"].mean(),
        "平均局部起伏度": seg["local_relief_3x3"].mean(),
        "平均粗糙度": seg["roughness_3x3"].mean(),
        "平均距一级分水岭(m)": seg["dist_to_ridge1"].mean(),
        "平均距河网(m)": seg["dist_to_river"].mean()
    })
segment_df = pd.DataFrame(segment_rows)


# ============================================================
# 20. 图件
# ============================================================
log_step("开始绘制图件...")
extent = [x_sub.min(), x_sub.max(), y_sub.min(), y_sub.max()]
extent_c = [x_c.min(), x_c.max(), y_c.min(), y_c.max()]

# 图1
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(z_sub, extent=extent, origin="lower", aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Elevation (m)")
ax.plot(river["x"], river["y"], linestyle="None", marker=".", markersize=1.2, alpha=0.35, label="河网")
ax.plot(ridge1["x"], ridge1["y"], linewidth=1.0, alpha=0.75, label="一级分水岭")
ax.plot(road_base["x"], road_base["y"], linewidth=1.6, alpha=0.90, label="原秦直道")
ax.plot(new_route["x"], new_route["y"], linewidth=2.4, color="red", label="重规划路线")
if len(related_sites) > 0:
    ax.scatter(related_sites["x"], related_sites["y"], s=55, marker="x", label="相关遗存")
ax.scatter(start_xy[0], start_xy[1], s=85, marker="o", label="起点")
ax.scatter(end_xy[0], end_xy[1], s=95, marker="^", label="终点")
ax.set_title("第三问：DEM背景下的新旧路线对比")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG1, dpi=300, bbox_inches="tight")
plt.close()

# 图2
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(cost_surface, extent=extent_c, origin="lower", aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("综合代价")
ax.plot(road_base["x"], road_base["y"], linewidth=1.2, alpha=0.7, label="原秦直道")
ax.plot(new_route["x"], new_route["y"], linewidth=2.3, color="red", label="重规划路线")
ax.plot(ridge1["x"], ridge1["y"], linewidth=0.8, alpha=0.55, label="一级分水岭")
ax.set_title("第三问：综合代价面与重规划路径")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG2, dpi=300, bbox_inches="tight")
plt.close()

# 图3
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(mobility_df))
width = 0.38
ax.bar(x - width/2, mobility_df["原秦直道"], width, label="原秦直道")
ax.bar(x + width/2, mobility_df["重规划路线"], width, label="重规划路线")
ax.set_xticks(x)
ax.set_xticklabels(mobility_df["指标"], rotation=20)
ax.set_title("第三问：通行性关键指标对比")
ax.legend()
plt.tight_layout()
plt.savefig(FIG3, dpi=300, bbox_inches="tight")
plt.close()

# 图4
fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)
axes[0].plot(old_feat["cumdist"], old_feat["elevation"], label="原秦直道")
axes[0].plot(new_feat["cumdist"], new_feat["elevation"], label="重规划路线")
axes[0].set_title("高程剖面对比")
axes[0].set_ylabel("Elevation (m)")
axes[0].legend()

axes[1].plot(old_feat["cumdist"], old_feat["slope_deg"], label="原秦直道")
axes[1].plot(new_feat["cumdist"], new_feat["slope_deg"], label="重规划路线")
axes[1].set_title("坡度剖面对比")
axes[1].set_ylabel("Slope (°)")
axes[1].legend()

axes[2].plot(old_feat["cumdist"], old_feat["dist_to_ridge1"], label="原秦直道")
axes[2].plot(new_feat["cumdist"], new_feat["dist_to_ridge1"], label="重规划路线")
axes[2].set_title("距一级分水岭距离剖面对比")
axes[2].set_ylabel("Distance to ridge1 (m)")
axes[2].legend()

axes[3].plot(old_feat["cumdist"], old_feat["dist_to_river"], label="原秦直道")
axes[3].plot(new_feat["cumdist"], new_feat["dist_to_river"], label="重规划路线")
axes[3].set_title("距河网距离剖面对比")
axes[3].set_ylabel("Distance to river (m)")
axes[3].set_xlabel("累计距离 (m)")
axes[3].legend()

plt.tight_layout()
plt.savefig(FIG4, dpi=300, bbox_inches="tight")
plt.close()

# 图5
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(z_sub, extent=extent, origin="lower", aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Elevation (m)")
ax.plot(road_base["x"], road_base["y"], linewidth=1.2, alpha=0.65, label="原秦直道")
ax.plot(new_route["x"], new_route["y"], linewidth=2.0, color="red", label="重规划路线")

for idx, row in segment_df.iterrows():
    xs = [row["起点x"], row["终点x"]]
    ys = [row["起点y"], row["终点y"]]
    ax.plot(xs, ys, linewidth=4.0, alpha=0.9, label="偏移较大路段" if idx == 0 else None)

ax.set_title("第三问：局部偏移较大路段识别")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(FIG5, dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 21. 保存结果
# ============================================================
log_step("开始保存 Excel 与图件...")
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    new_route.to_excel(writer, sheet_name="重规划路线坐标", index=False)
    compare_df.to_excel(writer, sheet_name="新旧路线总对比", index=False)
    mobility_df.to_excel(writer, sheet_name="通行性优化表", index=False)
    spatial_shift_df.to_excel(writer, sheet_name="空间偏移表", index=False)
    param_df.to_excel(writer, sheet_name="数据驱动参数标定", index=False)
    old_feat.to_excel(writer, sheet_name="原路线特征", index=False)
    new_feat.to_excel(writer, sheet_name="重规划路线特征", index=False)
    segment_df.to_excel(writer, sheet_name="局部偏移路段分析", index=False)

print("\n已保存:", OUT_XLSX)
print("已保存图片：")
print(FIG1)
print(FIG2)
print(FIG3)
print(FIG4)
print(FIG5)

print("\n" + "="*72)
print("第三问量化总结")
print("="*72)
for _, row in mobility_df.iterrows():
    print(f"{row['指标']}：改善 {row['绝对改善量']:.2f}，改善率 {row['改善率(%)']:.2f}%")

ridge_shift = spatial_shift_df.loc[spatial_shift_df["指标"] == "平均距一级分水岭变化(m)", "变化量(新-旧)"].values[0]
river_shift = spatial_shift_df.loc[spatial_shift_df["指标"] == "平均距河网变化(m)", "变化量(新-旧)"].values[0]

print(f"平均距一级分水岭变化量：{ridge_shift:.2f} m")
print(f"平均距河网变化量：{river_shift:.2f} m")

if related_tree is not None:
    related_shift = spatial_shift_df.loc[spatial_shift_df["指标"] == "平均距相关遗存变化(m)", "变化量(新-旧)"].values[0]
    print(f"平均距相关遗存变化量：{related_shift:.2f} m")

print("说明：")
print("1) 第三问以宏观地形骨架稳定为前提，用现代 DEM 反推古代宏观选线规律。")
print("2) 起终点直接取原路线首尾点，不再采用最长连续有效段思路。")
print("3) 候选区只保留 DEM 有效区、极端坡度和可选遗存避让等真正硬约束。")
print("4) 分水岭依附、历史走廊一致性、河网规避全部作为软代价参与优化。")
print("5) 代码已加入关键阶段日志与进度条，可实时查看程序推进情况。")
print("="*72)