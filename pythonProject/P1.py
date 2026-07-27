#下列代码展示了为什么一直以来都在数据处理上面出错，文件陕甘八县的高程数据.csv的y轴从上到下递增，在处理数据时要反过来
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEM_CSV = "陕甘八县的高程数据.csv"
ATT3_XLSX = "附件3  甘陕八县的县界数据.xlsx"

assert os.path.exists(DEM_CSV), f"未找到文件: {DEM_CSV}"
assert os.path.exists(ATT3_XLSX), f"未找到文件: {ATT3_XLSX}"

# 降采样倍数：只是看方向，8 或 10 都可以
STEP = 8


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
        cand = [c for c in cols if "x" in str(c).lower()]
        if not cand:
            raise ValueError(f"无法识别 x 列: {cols}")
        xcol = cand[0]

    if ycol is None:
        cand = [c for c in cols if "y" in str(c).lower()]
        if not cand:
            raise ValueError(f"无法识别 y 列: {cols}")
        ycol = cand[0]

    return df.rename(columns={xcol: "x", ycol: "y"}).copy()


def coord_to_edges(arr):
    arr = np.asarray(arr, dtype=float)
    d = np.diff(arr)
    if not np.all(d > 0):
        raise ValueError("坐标必须严格递增")
    edges = np.empty(len(arr) + 1, dtype=float)
    edges[1:-1] = (arr[:-1] + arr[1:]) / 2
    edges[0] = arr[0] - d[0] / 2
    edges[-1] = arr[-1] + d[-1] / 2
    return edges


main_bar = tqdm(total=5, desc="总进度")

# ============================================================
# 1. 读取 DEM
# ============================================================
print("Step 1/5: 读取 DEM")
dem_raw = pd.read_csv(DEM_CSV, header=None)

x_coords = pd.to_numeric(dem_raw.iloc[0, 1:], errors="coerce").to_numpy(dtype=float)
y_coords = pd.to_numeric(dem_raw.iloc[1:, 0], errors="coerce").to_numpy(dtype=float)

z = dem_raw.iloc[1:, 1:].replace("NA", np.nan)
z = z.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

print("x 是否递增:", np.all(np.diff(x_coords) > 0))
print("y 是否递增:", np.all(np.diff(y_coords) > 0))
print("DEM 原始形状:", z.shape)

if not np.all(np.diff(x_coords) > 0):
    raise ValueError("x_coords 不是严格递增")
if not np.all(np.diff(y_coords) > 0):
    raise ValueError("当前代码要求 y_coords 严格递增")

main_bar.update(1)

# ============================================================
# 2. 降采样 + 构造掩膜
# ============================================================
print("Step 2/5: 降采样并构造有效掩膜")
x_coords_ds = x_coords[::STEP]
y_coords_ds = y_coords[::STEP]
z_ds = z[::STEP, ::STEP]

valid_mask = np.isfinite(z_ds).astype(float)
valid_mask[valid_mask == 0] = np.nan

x_edges = coord_to_edges(x_coords_ds)
y_edges = coord_to_edges(y_coords_ds)

mask_raw = valid_mask
mask_flip_ud = np.flipud(valid_mask)
mask_flip_lr = np.fliplr(valid_mask)

print("降采样后形状:", z_ds.shape)
print("有效栅格数(降采样后):", int(np.isfinite(z_ds).sum()))
print("无效栅格数(降采样后):", int(np.isnan(z_ds).sum()))

main_bar.update(1)

# ============================================================
# 3. 读取八县边界
# ============================================================
print("Step 3/5: 读取八县边界")
boundary_xls = pd.ExcelFile(ATT3_XLSX)
boundary_data = []

for sheet in tqdm(boundary_xls.sheet_names, desc="读取县界"):
    df = pd.read_excel(ATT3_XLSX, sheet_name=sheet)
    df = standardize_xy(df)
    boundary_data.append((sheet, df))

main_bar.update(1)

# ============================================================
# 4. 绘制三张方向对照图
# ============================================================
print("Step 4/5: 绘制方向诊断图")
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

titles = ["原始掩膜", "上下翻转后", "左右翻转后"]
masks = [mask_raw, mask_flip_ud, mask_flip_lr]

for ax, title, mask in tqdm(list(zip(axes, titles, masks)), desc="绘制子图", total=3):
    ax.pcolormesh(
        x_edges, y_edges, mask,
        cmap="Greens",
        shading="auto",
        alpha=0.65
    )

    for sheet, df in boundary_data:
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        ax.plot(x, y, color="black", linewidth=1.0)

        if (x[0] != x[-1]) or (y[0] != y[-1]):
            ax.plot([x[-1], x[0]], [y[-1], y[0]], color="black", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("x坐标/m")
    ax.set_ylabel("y坐标/m")
    ax.set_aspect("equal")
    ax.grid(False)

plt.tight_layout()
main_bar.update(1)

# ============================================================
# 5. 只显示，不保存
# ============================================================
print("Step 5/5: 显示图片")
plt.show()
plt.close(fig)

main_bar.update(1)
main_bar.close()

# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from tqdm.auto import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 0. 参数区（相对路径）
# ============================================================
DEM_CSV = "陕甘八县的高程数据.csv"
ATT2_XLSX = "附件2  秦直道及周边地形和相关遗迹的数据.xlsx"
ATT3_XLSX = "附件3  甘陕八县的县界数据.xlsx"   # 仅用于总览图

SAVE_OVERVIEW = True
OVERVIEW_DPI = 160
OVERVIEW_DOWNSAMPLE = 8

TARGET_BG_N = 3000          # 背景点数量
BUFFER_INNER = 300.0        # 距秦直道下限
BUFFER_OUTER = 5000.0       # 距秦直道上限
BG_MAX_ROUNDS = 30
BG_PER_ROUND = 150000

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


def split_polyline_by_seq_reset(df):
    """
    正确处理“河网”“二级分水岭”：
    每一段内部序号都从1开始，因此按“序号回跳”切段。
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
    parts = split_polyline_by_seq_reset(df)
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
                _, one_dist = tree.query_nearest([pt], return_distance=True, all_matches=False)
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

        win = z[r0:r1, c0:c1]
        vals = win[~np.isnan(win)]
        center = z[iy, ix]

        if len(vals) == 0 or np.isnan(center):
            relief[i] = np.nan
            tpi[i] = np.nan
            rough[i] = np.nan
        else:
            relief[i] = np.max(vals) - np.min(vals)
            tpi[i] = center - np.mean(vals)
            rough[i] = np.std(vals)

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


# ============================================================
# 2. 读取 DEM（修正版：强制翻转 z，不翻 y_coords）
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

# 关键修正：仅翻转 z
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

print("\n[遗迹类型统计]")
print(sites["类型"].value_counts(dropna=False))

beacon = sites[sites["类型"].astype(str).str.contains("烽火", na=False)].copy()
pass_site = sites[sites["类型"].astype(str).str.contains("关", na=False)].copy()
other_sites = sites[~sites.index.isin(beacon.index) & ~sites.index.isin(pass_site.index)].copy()


# ============================================================
# 4. 建立线段空间索引
# ============================================================
print("\n[建立线段索引中...]")
road_tree, road_segs, road_seg_angles = build_segment_index_single(road, name="秦直道")
ridge1_tree, ridge1_segs, ridge1_seg_angles = build_segment_index_single(ridge1, name="一级分水岭")
ridge2_tree, ridge2_segs, ridge2_seg_angles = build_segment_index_multi(ridge2, name="二级分水岭")
river_tree, river_segs, river_seg_angles = build_segment_index_multi(river, name="河网")

print("秦直道线段数:", len(road_segs))
print("一级分水岭线段数:", len(ridge1_segs))
print("二级分水岭线段数:", len(ridge2_segs))
print("河网线段数:", len(river_segs))

if len(road_segs) == 0 or len(ridge1_segs) == 0:
    raise RuntimeError("建线段失败，请检查附件2数据。")


# ============================================================
# 5. 总览图
# ============================================================
if SAVE_OVERVIEW:
    print("\n[生成总览图中...]")
    step = max(1, int(OVERVIEW_DOWNSAMPLE))
    z_plot = z[::step, ::step]

    plt.figure(figsize=(12, 10))
    plt.imshow(
        z_plot,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="auto",
        cmap="terrain",
        alpha=0.85
    )
    plt.colorbar(label="Elevation (m)")

    plt.plot(road["x"], road["y"], color="red", linewidth=1.6, label="秦直道")
    plt.plot(ridge1["x"], ridge1["y"], color="green", linewidth=1.0, label="一级分水岭")

    ws2_parts = split_polyline_by_seq_reset(ridge2)
    first = True
    for seg in ws2_parts:
        plt.plot(seg["x"], seg["y"], color="purple", linewidth=1.0,
                 label="二级分水岭" if first else None)
        first = False

    river_parts = split_polyline_by_seq_reset(river)
    first = True
    for seg in river_parts:
        plt.plot(seg["x"], seg["y"], color="#87CEFA", linewidth=0.7,
                 label="河网" if first else None)
        first = False

    if len(pass_site) > 0:
        plt.scatter(pass_site["x"], pass_site["y"], s=50, marker="s",
                    color="gold", edgecolors="black", linewidths=0.4, label="沿线关隘")
    if len(beacon) > 0:
        plt.scatter(beacon["x"], beacon["y"], s=45, marker="^",
                    color="darkorange", edgecolors="black", linewidths=0.4, label="沿线烽火台")
    if len(other_sites) > 0:
        plt.scatter(other_sites["x"], other_sites["y"], s=25, marker="o",
                    color="magenta", edgecolors="black", linewidths=0.3, label="相关遗存")

    plt.title("问题1：DEM + 秦直道 + 分水岭 + 河网 + 遗迹")
    plt.xlabel("x坐标/m")
    plt.ylabel("y坐标/m")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig("问题1_总览图.png", dpi=OVERVIEW_DPI, bbox_inches="tight")
    plt.close()
    print("已保存: 问题1_总览图.png")


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
# 7. 特征提取函数
# ============================================================
road["line_angle"] = compute_line_angles(road)

def extract_features(points_df, name="points", has_line_angle=False):
    print(f"\n开始提取 [{name}] 特征...")
    df = points_df.copy().reset_index(drop=True)
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)

    print(f"[{name}] 1/5 栅格特征插值中...")
    df["elevation"] = sample_interp(interp_elev, xs, ys)
    df["slope_deg"] = sample_interp(interp_slope, xs, ys)
    df["aspect_deg"] = sample_interp(interp_aspect, xs, ys)

    relief_vals, tpi_vals, rough_vals = local_window_features(
        z, x_coords, y_coords, xs, ys, desc=f"{name}-3x3局部窗口特征"
    )
    df["local_relief_3x3"] = relief_vals
    df["tpi_3x3"] = tpi_vals
    df["roughness_3x3"] = rough_vals

    print(f"[{name}] 2/5 线距离计算中...")
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

    print(f"[{name}] 3/5 点距离计算中...")
    df["dist_to_sites"] = nearest_point_distance(xs, ys, sites, desc=f"{name}-到遗迹点")
    df["dist_to_beacon"] = nearest_point_distance(xs, ys, beacon, desc=f"{name}-到烽火台")
    df["dist_to_pass"] = nearest_point_distance(xs, ys, pass_site, desc=f"{name}-到关隘")

    print(f"[{name}] 4/5 DEM 有效性检查中...")
    in_extent, in_valid = check_points_status_by_interp(xs, ys, xmin, xmax, ymin, ymax, interp_elev)
    df["in_dem_extent"] = in_extent
    df["in_dem_valid_mask"] = in_valid

    print(f"[{name}] 5/5 方向夹角计算中...")
    if has_line_angle and "line_angle" in df.columns:
        nearest_r1_angle = query_nearest_segment_angle(
            ridge1_tree, ridge1_seg_angles, xs, ys,
            desc=f"{name}-最近一级分水岭方向"
        )
        df["angle_diff_to_ridge1"] = angle_diff_deg(
            df["line_angle"].to_numpy(dtype=float), nearest_r1_angle
        )
    else:
        df["angle_diff_to_ridge1"] = np.nan

    print(f"[{name}] 提取完成，共 {len(df)} 个点")
    return df


# ============================================================
# 8. 秦直道全线特征
# ============================================================
road_feat = extract_features(road, name="秦直道全线", has_line_angle=True)

road_feat_valid = road_feat[road_feat["in_dem_valid_mask"]].copy().reset_index(drop=True)
road_feat_invalid = road_feat[~road_feat["in_dem_valid_mask"]].copy().reset_index(drop=True)

print("\n[秦直道有效/无效点统计]")
print("总点数:", len(road_feat))
print("有效点数:", len(road_feat_valid))
print("无效点数:", len(road_feat_invalid))

print("\n[秦直道有效点特征统计]")
print(road_feat_valid[[
    "elevation", "slope_deg", "local_relief_3x3",
    "tpi_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_ridge2", "dist_to_river",
    "angle_diff_to_ridge1"
]].describe())


# ============================================================
# 9. 题目给定 10 个点
# ============================================================
target_points = pd.DataFrame({
    "序号": [1,2,3,4,5,6,7,8,9,10],
    "类型": ["秦直道","秦直道","秦直道","秦直道","秦直道","秦直道","烽火台","烽火台","关隘","关隘"],
    "x": [1292176.07,1315893.15,1319911.01,1334988.77,1345509.95,
          1373110.96,1307404.10,1359078.89,1374526.53,1362751.20],
    "y": [4105424.08,4085747.84,4065228.58,4042973.91,4025746.98,
          3974301.37,4094344.62,4011143.96,3965855.59,3998089.80]
})

target_feat = extract_features(target_points, name="题目指定10个点", has_line_angle=False)

print("\n[10个点特征结果]")
print(target_feat)


# ============================================================
# 10. 构造道路周边背景点，并与秦直道对比
# ============================================================
valid_rc = np.argwhere(~np.isnan(z))
print("\n[有效 DEM 格点数]", len(valid_rc))

np.random.seed(42)
bg_list = []
per_round = min(BG_PER_ROUND, len(valid_rc))

for rd in tqdm(range(BG_MAX_ROUNDS), desc="背景点抽样轮次"):
    sel = np.random.choice(len(valid_rc), size=per_round, replace=False)
    rc = valid_rc[sel]

    xs = x_coords[rc[:, 1]]
    ys = y_coords[rc[:, 0]]

    d = query_nearest_segment_distance(
        road_tree, xs, ys,
        desc=f"第{rd+1}轮背景点-到秦直道距离"
    )
    keep = (d >= BUFFER_INNER) & (d <= BUFFER_OUTER)

    if np.any(keep):
        sub = pd.DataFrame({"x": xs[keep], "y": ys[keep]})
        bg_list.append(sub)

    curr_n = sum(len(t) for t in bg_list)
    tqdm.write(f"背景点抽样轮次 {rd+1}/{BG_MAX_ROUNDS}，当前候选数 = {curr_n}")
    if curr_n >= TARGET_BG_N:
        break

if len(bg_list) == 0:
    raise RuntimeError("未能抽取到足够背景点，请增大 BUFFER_OUTER 或 BG_MAX_ROUNDS。")

bg_points = pd.concat(bg_list, ignore_index=True).drop_duplicates()
if len(bg_points) > TARGET_BG_N:
    bg_points = bg_points.sample(TARGET_BG_N, random_state=42).reset_index(drop=True)

print("\n[最终背景点数量]", len(bg_points))

bg_feat = extract_features(bg_points, name="道路缓冲带背景点", has_line_angle=False)
bg_feat_valid = bg_feat[bg_feat["in_dem_valid_mask"]].copy().reset_index(drop=True)

compare_cols = [
    "elevation", "slope_deg", "local_relief_3x3",
    "tpi_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_ridge2", "dist_to_river"
]

compare_summary = pd.DataFrame({
    "road_mean": road_feat_valid[compare_cols].mean(),
    "buffer_bg_mean": bg_feat_valid[compare_cols].mean(),
    "road_median": road_feat_valid[compare_cols].median(),
    "buffer_bg_median": bg_feat_valid[compare_cols].median(),
})
compare_summary["mean_diff(road-bg)"] = compare_summary["road_mean"] - compare_summary["buffer_bg_mean"]
compare_summary["median_diff(road-bg)"] = compare_summary["road_median"] - compare_summary["buffer_bg_median"]

print("\n[秦直道有效点 vs 背景点 对比]")
print(compare_summary)


# ============================================================
# 11. 论文可用汇总表
# ============================================================
paper_summary = pd.DataFrame({
    "特征": compare_cols,
    "秦直道均值": [road_feat_valid[c].mean() for c in compare_cols],
    "背景点均值": [bg_feat_valid[c].mean() for c in compare_cols],
    "秦直道中位数": [road_feat_valid[c].median() for c in compare_cols],
    "背景点中位数": [bg_feat_valid[c].median() for c in compare_cols],
})

paper_summary2 = pd.DataFrame({
    "特征": ["angle_diff_to_ridge1"],
    "秦直道均值": [road_feat_valid["angle_diff_to_ridge1"].mean()],
    "背景点均值": [np.nan],
    "秦直道中位数": [road_feat_valid["angle_diff_to_ridge1"].median()],
    "背景点中位数": [np.nan],
})

paper_summary = pd.concat([paper_summary, paper_summary2], ignore_index=True)


# ============================================================
# 12. 导出 result1.xlsx
# ============================================================
road_feat_export = road_feat_valid.copy().reset_index(drop=True)
road_feat_export.insert(0, "序号", np.arange(1, len(road_feat_export) + 1))

result1_export = road_feat_export[[
    "序号", "x", "y",
    "elevation", "slope_deg", "aspect_deg",
    "local_relief_3x3", "tpi_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_ridge2", "dist_to_river",
    "dist_to_road", "dist_to_sites", "dist_to_beacon", "dist_to_pass",
    "angle_diff_to_ridge1",
    "in_dem_extent", "in_dem_valid_mask"
]].copy()

result1_export = result1_export.rename(columns={
    "x": "x坐标/m",
    "y": "y坐标/m"
})

invalid_export = road_feat_invalid.copy().reset_index(drop=True)
invalid_export.insert(0, "序号", np.arange(1, len(invalid_export) + 1))
invalid_export = invalid_export.rename(columns={
    "x": "x坐标/m",
    "y": "y坐标/m"
})

target_export = target_feat[[
    "序号", "类型", "x", "y",
    "elevation", "slope_deg", "aspect_deg",
    "local_relief_3x3", "tpi_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_ridge2", "dist_to_river",
    "dist_to_road", "dist_to_sites", "dist_to_beacon", "dist_to_pass",
    "angle_diff_to_ridge1",
    "in_dem_extent", "in_dem_valid_mask"
]].copy()

target_export = target_export.rename(columns={
    "x": "x坐标/m",
    "y": "y坐标/m"
})

bg_export = bg_feat_valid[[
    "x", "y",
    "elevation", "slope_deg", "aspect_deg",
    "local_relief_3x3", "tpi_3x3", "roughness_3x3",
    "dist_to_ridge1", "dist_to_ridge2", "dist_to_river",
    "dist_to_road", "dist_to_sites", "dist_to_beacon", "dist_to_pass",
    "in_dem_extent", "in_dem_valid_mask"
]].copy()

bg_export = bg_export.rename(columns={
    "x": "x坐标/m",
    "y": "y坐标/m"
})

save_path = "result1.xlsx"
with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    result1_export.to_excel(writer, sheet_name="result1", index=False)
    invalid_export.to_excel(writer, sheet_name="秦直道无效点", index=False)
    target_export.to_excel(writer, sheet_name="题目给定10点特征", index=False)
    bg_export.to_excel(writer, sheet_name="缓冲带背景点特征", index=False)
    compare_summary.to_excel(writer, sheet_name="秦直道与缓冲背景对比")
    paper_summary.to_excel(writer, sheet_name="论文可用汇总", index=False)

print("\n已保存:", save_path)
if SAVE_OVERVIEW:
    print("已保存: 问题1_总览图.png")