#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二赛段只读视觉调试网页。

订阅左右鱼眼、前向 RGB 与深度，在后台按 ``stage2_node`` 的检测流程重算一遍，
把「通过 / 未通过」的候选、逐条参数判定和当前生效的阈值画到图上，
再发布到本机 HTTP 页面。

覆盖的检测：

* 鱼眼相机：橙球 + 蓝球（HSV + 圆度/宽高比/填充率），并标注侧撞子链的
  入口居中窗口、撞击半径与接近 vx 计算；
* RGB 相机：橙球 + 蓝球（HSV + 面积 + 深度取样）与黄色停止线
  （ROI + HSV + 横线三条件），并标注中线对齐、左侧近球避让、
  各巡航子阶段的黄线触发/减速比例。

本节点不创建 ``Robot_Ctrl``，不发布任何运动命令，可以在真机上通过
SSH 端口转发安全使用。

两点与 ``stage2_node`` 的差异，页面上也有标注：

* ``stage2_node`` 的鱼眼只检测橙球（侧撞只打橙球），蓝球检测是本工具
  额外加的观察项，用 ``fisheye_blue_min_contour_area`` 单独控制；
* 黄线的「前方横线」三条件在 ``stage2_node`` 里按状态开关，本工具同时
  给出 strict（要求横线）和 loose（只要面积）两套选择结果。
"""

from __future__ import print_function

import argparse
import collections
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlsplit

# Support direct source-tree execution:
# python3 control_node/stage2_vision_preview.py ...
if __package__ is None or __package__ == '':
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402


STREAMS = collections.OrderedDict([
    ('fisheye_left', {
        'label': '左鱼眼：橙球 / 蓝球（PASS+FAIL）', 'primary': True,
        'withPanel': True}),
    ('fisheye_right', {
        'label': '右鱼眼：橙球 / 蓝球（PASS+FAIL）', 'primary': True,
        'withPanel': True}),
    ('fisheye_masks', {
        'label': '鱼眼颜色掩膜（左右 × 橙蓝）', 'primary': True}),
    ('rgb_balls', {
        'label': 'RGB 橙球 / 蓝球 + 中线对齐', 'primary': True,
        'withPanel': True}),
    ('rgb_yellow', {
        'label': 'RGB 黄线（strict + loose）', 'primary': True,
        'withPanel': True}),
    ('rgb_masks', {
        'label': 'RGB 颜色掩膜（橙 / 蓝 / 黄 ROI）', 'primary': True}),
    ('depth', {'label': '深度图（球深度取样来源）', 'primary': True}),
])


PAGE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyberDog 第二赛段视觉调试</title>
<style>
:root{color-scheme:dark;--bg:#08111e;--panel:#111d2e;--line:#29384d;
--text:#edf4fc;--muted:#91a5bd;--ok:#35d383;--warn:#ffc857;--bad:#ff6b7e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;gap:14px;align-items:center;
justify-content:space-between;padding:13px 18px;background:rgba(8,17,30,.95);
border-bottom:1px solid var(--line)}
.title{font-size:20px;font-weight:760}
.subtitle{color:var(--muted);font-size:12px}
.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.pill,select,button{border:1px solid var(--line);border-radius:8px;
background:#17253a;color:var(--text);padding:6px 10px}
.pill{color:var(--muted)}button{cursor:pointer}
main{max-width:1800px;margin:auto;padding:14px}
.notice{margin-bottom:12px;padding:10px 12px;border:1px solid #6a5221;
border-radius:9px;background:#2b2415;color:#ffd878}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.card{min-width:0;background:var(--panel);border:1px solid var(--line);
border-radius:12px;overflow:hidden}
.head{display:flex;align-items:center;justify-content:space-between;gap:8px;
padding:9px 12px;border-bottom:1px solid var(--line)}
.name{font-weight:700}
.health{color:var(--warn)}.health.ok{color:var(--ok)}.health.bad{color:var(--bad)}
.viewport{position:relative;aspect-ratio:4/3;background:#02060b;display:grid;
place-items:center}
.card.with-panel .viewport{aspect-ratio:1/1.05}
.viewport img{display:block;width:100%;height:100%;object-fit:contain}
.placeholder{position:absolute;inset:0;display:grid;place-items:center;
color:var(--muted);background:rgba(2,6,11,.75)}
.placeholder[hidden]{display:none}
.meta{padding:8px 12px;color:var(--muted);font-size:12px}
.details{margin-top:12px;display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.detail{padding:10px 12px;background:var(--panel);border:1px solid var(--line);
border-radius:10px}
.detail b{display:block;margin-bottom:4px}
.detail pre{margin:0;color:var(--muted);white-space:pre-wrap;
word-break:break-word;font:12px/1.45 ui-monospace,monospace}
footer{max-width:1800px;margin:auto;padding:0 16px 18px;color:var(--muted);
font-size:12px}
@media(max-width:1200px){.grid{grid-template-columns:1fr}
.details{grid-template-columns:1fr}header{align-items:flex-start;
flex-direction:column}}
</style>
</head>
<body>
<header>
<div>
<div class="title">CyberDog 第二赛段视觉调试</div>
<div class="subtitle">只订阅相机 · 不发送运动命令 · 适合 SSH 端口转发</div>
</div>
<div class="controls">
<span id="summary" class="pill">正在连接…</span>
<label>刷新 <select id="rate">
<option selected>1</option><option>2</option><option>3</option><option>5</option>
</select> FPS</label>
<button id="pause">暂停</button>
</div>
</header>
<main>
<div id="notice" class="notice" hidden></div>
<section id="grid" class="grid"></section>
<section class="details">
<div class="detail"><b>相机</b><pre id="camera">等待数据</pre></div>
<div class="detail"><b>鱼眼侧撞判定</b><pre id="fisheye">等待数据</pre></div>
<div class="detail"><b>RGB 球与中线</b><pre id="balls">等待数据</pre></div>
<div class="detail"><b>黄线</b><pre id="yellow">等待数据</pre></div>
</section>
</main>
<footer>
绿色框 = 通过全部参数，红色框 = 被某条参数过滤（面板里带 - 的就是失败项）。
鱼眼图上的白色虚框是侧撞入口居中窗口，白色圆环是撞击半径阈值；
RGB 图上的绿色竖线是两球中点，黄色横线是各子阶段的黄线触发/减速比例。
页面切到后台会自动暂停拉图。
</footer>
<script>
'use strict';
const defs = __STREAM_DEFINITIONS__;
const app = {paused: false, hidden: document.hidden,
             period: __CLIENT_PERIOD_MS__, statusFails: 0};
const grid = document.getElementById('grid');
const summary = document.getElementById('summary');
const notice = document.getElementById('notice');
const rate = document.getElementById('rate');
const pause = document.getElementById('pause');
const cards = defs.map(d => {
  const e = document.createElement('article');
  e.className = 'card' + (d.withPanel ? ' with-panel' : '');
  e.innerHTML = '<div class="head"><span class="name">' + d.label +
    '</span><span class="health">等待首帧</span></div>' +
    '<div class="viewport"><img alt="' + d.label +
    '"><div class="placeholder">等待首帧…</div></div>' +
    '<div class="meta">帧号 <span class="seq">0</span> · 预览延迟 ' +
    '<span class="age">-</span></div>';
  grid.appendChild(e);
  return {key: d.key, e: e, img: e.querySelector('img'),
          health: e.querySelector('.health'),
          placeholder: e.querySelector('.placeholder'),
          seq: e.querySelector('.seq'), age: e.querySelector('.age'),
          last: null, url: null};
});
const sleep = ms => new Promise(r => setTimeout(r, ms));
const stopped = () => app.paused || app.hidden;
async function get(url, ms) {
  const c = new AbortController();
  const t = setTimeout(() => c.abort(), ms);
  try {
    return await fetch(url, {cache: 'no-store', signal: c.signal});
  } finally {
    clearTimeout(t);
  }
}
function text(id, value) {
  document.getElementById(id).textContent = value;
}
async function statusLoop() {
  for (;;) {
    if (stopped()) { await sleep(200); continue; }
    try {
      const r = await get('/api/status', 1800);
      if (!r.ok) throw Error(r.status);
      const s = await r.json();
      app.statusFails = 0;
      summary.textContent = s.summary;
      summary.style.color = s.health === 'live' ? 'var(--ok)' :
        (s.health === 'error' ? 'var(--bad)' : 'var(--warn)');
      notice.hidden = !s.warning;
      notice.textContent = s.warning || '';
      text('camera', s.camera_text);
      text('fisheye', s.fisheye_text);
      text('balls', s.balls_text);
      text('yellow', s.yellow_text);
    } catch (e) {
      app.statusFails++;
      if (app.statusFails >= 3) {
        summary.textContent = '调试服务连接异常';
        summary.style.color = 'var(--bad)';
      }
    }
    await sleep(1000);
  }
}
async function show(card, blob) {
  const next = URL.createObjectURL(blob);
  const old = card.url;
  await new Promise((ok, bad) => {
    card.img.onload = ok;
    card.img.onerror = bad;
    card.img.src = next;
  });
  card.url = next;
  if (old) URL.revokeObjectURL(old);
}
async function frameLoop(card, delay) {
  await sleep(delay);
  for (;;) {
    if (stopped()) { await sleep(180); continue; }
    const started = performance.now();
    try {
      const q = card.last == null ? '' : '?after=' + card.last;
      const r = await get('/api/frame/' + card.key + q, 3500);
      if (r.status === 204) { await sleep(80); continue; }
      if (!r.ok) throw Error(r.status);
      await show(card, await r.blob());
      card.last = Number(r.headers.get('X-Frame-Id'));
      card.seq.textContent = card.last;
      card.age.textContent =
        Number(r.headers.get('X-Frame-Age') || 0).toFixed(2) + 's';
      card.health.textContent = '正常';
      card.health.className = 'health ok';
      card.placeholder.hidden = true;
    } catch (e) {
      card.health.textContent = '拉图失败';
      card.health.className = 'health bad';
    }
    await sleep(Math.max(60, app.period - (performance.now() - started)));
  }
}
rate.onchange = () => { app.period = 1000 / Number(rate.value); };
pause.onclick = () => {
  app.paused = !app.paused;
  pause.textContent = app.paused ? '恢复' : '暂停';
};
document.addEventListener('visibilitychange',
                          () => { app.hidden = document.hidden; });
window.addEventListener('beforeunload', () => cards.forEach(c => {
  if (c.url) URL.revokeObjectURL(c.url);
}));
statusLoop();
cards.forEach((c, i) => frameLoop(c, i * 90));
</script>
</body></html>'''


# ============================================================
# 参数默认值：名字与 stage2_node 完全一致，可直接复用同一个
# --params-file（例如 config/real_robot.yaml）。
# ============================================================
ORANGE_DEFAULTS = collections.OrderedDict([
    ('orange_h_min', 5),
    ('orange_h_max', 25),
    ('orange_s_min', 100),
    ('orange_s_max', 255),
    ('orange_v_min', 80),
    ('orange_v_max', 255),
    ('orange_min_contour_area', 400.0),
])

BLUE_DEFAULTS = collections.OrderedDict([
    ('blue_h_min', 90),
    ('blue_h_max', 130),
    ('blue_s_min', 80),
    ('blue_s_max', 255),
    ('blue_v_min', 50),
    ('blue_v_max', 255),
    ('blue_min_contour_area', 400.0),
])

FISHEYE_DEFAULTS = collections.OrderedDict([
    # 形状门（stage2_node.detect_fisheye_orange_ball）
    ('fisheye_orange_min_contour_area', 1000.0),
    # 只有本预览工具使用：stage2_node 的鱼眼不检测蓝球。
    ('fisheye_blue_min_contour_area', 1000.0),
    ('fisheye_morph_kernel_size', 5),
    ('fisheye_min_circularity', 0.60),
    ('fisheye_min_aspect_ratio', 0.65),
    ('fisheye_max_aspect_ratio', 1.45),
    ('fisheye_min_circle_fill_ratio', 0.50),
    ('fisheye_min_bbox_fill_ratio', 0.45),
    # 侧撞入口窗口
    ('fisheye_entry_center_x_ratio', 0.50),
    ('fisheye_entry_center_y_ratio', 0.50),
    ('fisheye_entry_x_tolerance', 0.18),
    ('fisheye_entry_y_tolerance', 0.25),
    ('fisheye_entry_confirm_frames', 3),
    # 接近 / 撞击
    ('fisheye_approach_target_x_ratio', 0.50),
    ('fisheye_approach_x_deadband_ratio', 0.05),
    ('fisheye_approach_vx_k', 0.60),
    ('fisheye_approach_vx_max', 0.15),
    ('fisheye_approach_vy', 0.20),
    ('fisheye_approach_lost_frames', 5),
    ('left_fisheye_x_to_vx_sign', 1.0),
    ('right_fisheye_x_to_vx_sign', -1.0),
    ('fisheye_hit_radius', 45.0),
    ('fisheye_hit_radius_confirm_frames', 3),
    ('fisheye_hit_vy', 0.30),
])

RGB_BALL_DEFAULTS = collections.OrderedDict([
    ('depth_search_half', 12),
    ('valid_min_depth_m', 0.05),
    ('valid_max_depth_m', 10.0),
    ('prefer_nearest_ball', True),
    ('lateral_align_px_tol', 20.0),
    ('center_ok_px', 10.0),
    ('center_cruise_fixed_vy', 0.10),
    ('center_depth_diff_disable_align_m', 0.50),
    ('center_far_side_fixed_vy', 0.03),
    ('stage2_left_ball_avoid_enabled', True),
    ('stage2_left_ball_avoid_center_px', 130),
    ('stage2_left_ball_avoid_depth_m', 0.45),
    ('stage2_left_ball_avoid_vy', 0.12),
    ('stage2_left_ball_avoid_confirm_frames', 2),
    ('stage2_left_ball_avoid_min_radius', 8.0),
])

YELLOW_DEFAULTS = collections.OrderedDict([
    ('yellow_roi_top_ratio', 0.65),
    ('yellow_roi_left_ratio', 0.4),
    ('yellow_roi_right_ratio', 0.6),
    ('yellow_h_min', 15),
    ('yellow_h_max', 40),
    ('yellow_s_min', 80),
    ('yellow_s_max', 255),
    ('yellow_v_min', 80),
    ('yellow_v_max', 255),
    ('yellow_min_contour_area', 100.0),
    ('yellow_min_width_height_ratio', 2.5),
    ('yellow_min_width_ratio', 0.45),
    ('yellow_center_tolerance_ratio', 0.15),
    # stage2_node 读取但没有参与过滤，这里只展示。
    ('yellow_max_tilt_deg', 30.0),
    ('yellow_stop_line_y_ratio_stage1', 1.0),
    ('yellow_stop_line_y_ratio_stage2', 1.0),
    ('yellow_stop_line_y_ratio_stage3', 1.0),
    ('yellow_ratio_scan', 0.9),
    ('yellow_ratio_final', 1.0),
    ('yellow_stop_confirm_count', 1),
    ('yellow_slowdown_ratio_stage1', 0.90),
    ('yellow_slowdown_ratio_stage2', 0.90),
    ('yellow_slowdown_ratio_stage3', 0.90),
    ('yellow_slowdown_ratio_scan', 0.80),
    ('yellow_slowdown_ratio_final', 0.90),
    ('yellow_angle_align_enabled', True),
    ('yellow_angle_align_fixed_wz', 0.15),
    ('yellow_angle_align_deadband_deg', 0.5),
])

# 黄线「前方横线」三条件在这些状态里不启用（stage2_node.detect_yellow_stop_line）。
YELLOW_LOOSE_STATES = (
    'STAGE1_CRUISE_BALL_AND_YELLOW',
    'STAGE3_CRUISE_BALL_ONLY',
    'STAGE3_GO_SCAN',
    'STAGE3_GO_FINAL',
)

ORANGE_BGR = (0, 140, 255)
BLUE_BGR = (255, 90, 0)
YELLOW_BGR = (0, 255, 255)
PASS_BGR = (60, 230, 60)
FAIL_BGR = (60, 60, 255)
WHITE_BGR = (255, 255, 255)
GREY_BGR = (190, 190, 190)


def _image_qos():
    return QoSProfile(
        history=qos_profile_sensor_data.history,
        depth=1,
        reliability=qos_profile_sensor_data.reliability,
        durability=qos_profile_sensor_data.durability,
    )


def _normalize_namespace(value):
    value = str(value).strip()
    if not value or value == '/':
        return ''
    return '/' + value.strip('/')


def _join_topic(namespace, suffix):
    return _normalize_namespace(namespace) + '/' + str(suffix).lstrip('/')


def _discover_namespace(node, timeout_sec):
    deadline = time.monotonic() + float(timeout_sec)
    preferred = ('stereo_camera', 'camera_server', 'camera')
    while time.monotonic() < deadline:
        try:
            pairs = node.get_node_names_and_namespaces()
        except Exception:
            pairs = []
        for wanted in preferred:
            for name, namespace in pairs:
                if name == wanted and namespace not in ('', '/'):
                    return _normalize_namespace(namespace)
        time.sleep(0.2)
    return ''


def _declare(node, defaults):
    for key, value in defaults.items():
        node.declare_parameter(key, value)


def _read(node, defaults):
    return collections.OrderedDict(
        (key, node.get_parameter(key).value) for key in defaults)


def _encode_jpeg(image, quality):
    ok, encoded = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError('OpenCV JPEG encoding failed')
    return encoded.tobytes()


def _put_label(image, text, point, color, scale=0.48):
    x, y = int(point[0]), int(point[1])
    cv2.putText(image, str(text), (x, max(14, y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, str(text), (x, max(14, y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _short_state(state):
    """Shorten a stage-2 cruise state name so it fits on the status panel."""
    return state.replace('_CRUISE_BALL_AND_YELLOW', '_CRUISE').replace(
        '_CRUISE_BALL_ONLY', '_CRUISE')


def _label_x(width, x, text, scale=0.40):
    """Keep a drawn label inside the image instead of clipping it."""
    estimated = int(len(str(text)) * 18.0 * scale)
    return max(3, min(int(x), max(3, width - estimated)))


def _condition(name, value, rule, passed):
    return {
        'name': str(name),
        'value': str(value),
        'rule': str(rule),
        'passed': bool(passed),
    }


def _accepted(record):
    return all(item['passed'] for item in record['checks'])


def _failed_names(record, limit=3):
    if record is None:
        return []
    return [item['name'] for item in record.get('checks', [])
            if not item['passed']][:limit]


def _draw_condition_panel(image, record, title, x, y, width):
    """Draw one candidate's per-parameter PASS/FAIL rows."""
    checks = record['checks'] if record is not None else []
    height = 50 + 15 * len(checks)
    x2 = min(image.shape[1] - 4, x + width)
    y2 = min(image.shape[0] - 4, y + height)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    cv2.rectangle(image, (x, y), (x2, y2), (65, 82, 105), 1)
    _put_label(image, title, (x + 7, y + 17), WHITE_BGR, 0.40)
    if record is None:
        _put_label(image, 'NO VISIBLE CANDIDATE', (x + 7, y + 37),
                   (0, 165, 255), 0.38)
        return
    color = PASS_BGR if record['accepted'] else FAIL_BGR
    _put_label(image, '%s  %s' % (
        'PASS' if record['accepted'] else 'FAIL', record.get('note', '')),
        (x + 7, y + 36), color, 0.40)
    line_y = y + 54
    for item in checks:
        item_color = PASS_BGR if item['passed'] else FAIL_BGR
        _put_label(image, '%s%-13s %-8s %s' % (
            '+' if item['passed'] else '-', item['name'],
            item['value'], item['rule']), (x + 6, line_y), item_color, 0.30)
        line_y += 15


def _append_status_panel(image, title, lines, panel_height):
    h, w = image.shape[:2]
    canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    canvas[:] = (8, 17, 30)
    canvas[:h, :w] = image
    cv2.line(canvas, (0, h), (w - 1, h), (65, 82, 105), 1)
    _put_label(canvas, title, (9, h + 20), WHITE_BGR, 0.44)
    line_y = h + 41
    for text, color in lines:
        _put_label(canvas, text, (9, line_y), color, 0.34)
        line_y += 17
    return canvas


def _append_condition_row(image, panels):
    """Append a bottom band holding up to three condition panels."""
    if not panels:
        return image
    h, w = image.shape[:2]
    rows = max(len(record['checks']) if record is not None else 0
               for _, record in panels)
    band = 62 + 15 * rows
    canvas = np.zeros((h + band, w, 3), dtype=np.uint8)
    canvas[:] = (8, 17, 30)
    canvas[:h, :w] = image
    cv2.line(canvas, (0, h), (w - 1, h), (65, 82, 105), 1)
    count = len(panels)
    width = max(150, (w - 8 - 6 * (count - 1)) // count)
    for index, (title, record) in enumerate(panels):
        _draw_condition_panel(canvas, record, title,
                              4 + index * (width + 6), h + 6, width)
    return canvas


def _mask_tile(mask, label, size):
    tile = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    _put_label(tile, label, (8, 20), YELLOW_BGR, 0.46)
    return tile


def _mosaic(tiles, columns):
    rows = [cv2.hconcat(tiles[index:index + columns])
            for index in range(0, len(tiles), columns)]
    return cv2.vconcat(rows) if len(rows) > 1 else rows[0]


def _blank_view(size, text):
    view = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    _put_label(view, text, (18, 34), FAIL_BGR, 0.66)
    return view


def _hsv_mask(frame, low, high, kernel_size):
    """HSV inRange + open + close, exactly like stage2_node."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8),
                       np.array(high, dtype=np.uint8))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _color_bounds(cfg, prefix):
    low = (int(cfg['%s_h_min' % prefix]), int(cfg['%s_s_min' % prefix]),
           int(cfg['%s_v_min' % prefix]))
    high = (int(cfg['%s_h_max' % prefix]), int(cfg['%s_s_max' % prefix]),
            int(cfg['%s_v_max' % prefix]))
    return low, high


def _hsv_text(cfg, prefix):
    low, high = _color_bounds(cfg, prefix)
    return 'H%d..%d S%d..%d V%d..%d' % (
        low[0], high[0], low[1], high[1], low[2], high[2])


def _contours(mask):
    return cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def _odd_kernel(value):
    size = max(1, int(value))
    return size + 1 if size % 2 == 0 else size


# ============================================================
# 鱼眼球检测（镜像 stage2_node.detect_fisheye_orange_ball，保留 FAIL）
# ============================================================
def fisheye_ball_records(frame, color_cfg, fisheye_cfg, color_name):
    """Mirror the fisheye ball filter and keep rejected contours.

    ``stage2_node`` only runs this for the orange ball; the blue variant uses
    the very same shape gates with the blue HSV range and is preview-only.
    """
    min_area = float(fisheye_cfg['fisheye_%s_min_contour_area' % color_name])
    kernel_size = _odd_kernel(fisheye_cfg['fisheye_morph_kernel_size'])
    low, high = _color_bounds(color_cfg, color_name)
    mask = _hsv_mask(frame, low, high, kernel_size)
    min_circularity = float(fisheye_cfg['fisheye_min_circularity'])
    min_aspect = float(fisheye_cfg['fisheye_min_aspect_ratio'])
    max_aspect = float(fisheye_cfg['fisheye_max_aspect_ratio'])
    min_circle_fill = float(fisheye_cfg['fisheye_min_circle_fill_ratio'])
    min_bbox_fill = float(fisheye_cfg['fisheye_min_bbox_fill_ratio'])
    display_area = max(25.0, min_area * 0.08)
    records = []
    for contour in _contours(mask):
        area = float(cv2.contourArea(contour))
        if area < display_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * math.pi * area / (perimeter * perimeter)
                       if perimeter > 1e-6 else 0.0)
        (cx, cy), enclosing_radius = cv2.minEnclosingCircle(contour)
        enclosing_radius = float(enclosing_radius)
        aspect_ratio = bw / float(max(bh, 1))
        circle_fill = area / max(math.pi * enclosing_radius ** 2, 1e-6)
        bbox_fill = area / float(max(bw * bh, 1))
        checks = [
            _condition('area', '%.0f' % area, '>=%.0f' % min_area,
                       area >= min_area),
            _condition('perimeter', '%.1f' % perimeter, '>0',
                       perimeter > 1e-6),
            _condition('radius', '%.1f' % enclosing_radius, '>0',
                       enclosing_radius > 1e-6),
            _condition('circularity', '%.3f' % circularity,
                       '>=%.2f' % min_circularity,
                       circularity >= min_circularity),
            _condition('aspect_ratio', '%.3f' % aspect_ratio,
                       '%.2f..%.2f' % (min_aspect, max_aspect),
                       min_aspect <= aspect_ratio <= max_aspect),
            _condition('circle_fill', '%.3f' % circle_fill,
                       '>=%.2f' % min_circle_fill,
                       circle_fill >= min_circle_fill),
            _condition('bbox_fill', '%.3f' % bbox_fill,
                       '>=%.2f' % min_bbox_fill, bbox_fill >= min_bbox_fill),
        ]
        radius = min(enclosing_radius, math.sqrt(area / math.pi))
        record = {
            'color': color_name,
            'center': (int(cx), int(cy)),
            'radius': float(radius),
            'radius_circle': enclosing_radius,
            'area': area,
            'bbox': (int(bx), int(by), int(bx + bw), int(by + bh)),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'circle_fill_ratio': float(circle_fill),
            'bbox_fill_ratio': float(bbox_fill),
            'image_shape': frame.shape[:2],
            'checks': checks,
        }
        record['accepted'] = _accepted(record)
        record['note'] = '%s r=%.1f area=%.0f' % (
            color_name, record['radius'], area)
        records.append(record)
    records.sort(key=lambda item: (
        item['accepted'],
        sum(check['passed'] for check in item['checks']),
        item['radius']), reverse=True)
    return records, mask


def fisheye_best(records):
    """stage2_node keeps the accepted candidate with the largest radius."""
    accepted = [item for item in records if item['accepted']]
    if not accepted:
        return None
    return max(accepted, key=lambda item: item['radius'])


def fisheye_entry_ok(target, cfg):
    """Mirror stage2_node.fisheye_target_near_center."""
    if target is None:
        return False, None, None
    h, w = target['image_shape']
    cx, cy = target['center']
    x_ratio = cx / float(max(w, 1))
    y_ratio = cy / float(max(h, 1))
    ok = (abs(x_ratio - float(cfg['fisheye_entry_center_x_ratio']))
          <= float(cfg['fisheye_entry_x_tolerance'])
          and abs(y_ratio - float(cfg['fisheye_entry_center_y_ratio']))
          <= float(cfg['fisheye_entry_y_tolerance']))
    return ok, x_ratio, y_ratio


def fisheye_approach_vx(target, cfg, side):
    """Mirror stage2_node.compute_fisheye_approach_vx."""
    if target is None:
        return 0.0, None
    _, w = target['image_shape']
    cx, _ = target['center']
    error = cx / float(max(w, 1)) - float(
        cfg['fisheye_approach_target_x_ratio'])
    if abs(error) <= float(cfg['fisheye_approach_x_deadband_ratio']):
        return 0.0, error
    sign = float(cfg['left_fisheye_x_to_vx_sign'] if side == 'left'
                 else cfg['right_fisheye_x_to_vx_sign'])
    limit = abs(float(cfg['fisheye_approach_vx_max']))
    value = sign * float(cfg['fisheye_approach_vx_k']) * error
    return max(-limit, min(limit, value)), error


# ============================================================
# RGB 球检测（镜像 stage2_node.detect_color_ball_candidates，保留 FAIL）
# ============================================================
def depth_sample(depth, encoding, rgb_center, rgb_shape, cfg):
    """Mirror stage2_node.get_depth_for_rgb_point."""
    if depth is None or not encoding:
        return None, None, None
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[:2]
    half = int(cfg['depth_search_half'])
    depth_cx = int(rgb_center[0] * dw / max(rw, 1))
    depth_cy = int(rgb_center[1] * dh / max(rh, 1))
    x1 = max(0, depth_cx - half)
    x2 = min(dw, depth_cx + half + 1)
    y1 = max(0, depth_cy - half)
    y2 = min(dh, depth_cy + half + 1)
    patch = depth[y1:y2, x1:x2]
    if encoding == '16UC1':
        patch_m = patch.astype(np.float32) / 1000.0
    elif encoding == '32FC1':
        patch_m = patch.astype(np.float32)
    else:
        return None, (depth_cx, depth_cy), (x1, y1, x2, y2)
    valid = patch_m[np.isfinite(patch_m)]
    valid = valid[(valid > float(cfg['valid_min_depth_m']))
                  & (valid < float(cfg['valid_max_depth_m']))]
    if valid.size == 0:
        return None, (depth_cx, depth_cy), (x1, y1, x2, y2)
    return float(np.percentile(valid, 20)), (depth_cx, depth_cy), \
        (x1, y1, x2, y2)


def rgb_ball_records(frame, depth, encoding, color_cfg, common_cfg,
                     color_name):
    """Mirror the RGB ball detector, keeping area/depth rejections visible."""
    h, w = frame.shape[:2]
    min_area = float(color_cfg['%s_min_contour_area' % color_name])
    low, high = _color_bounds(color_cfg, color_name)
    mask = _hsv_mask(frame, low, high, 5)
    image_center_x = w // 2
    display_area = max(20.0, min_area * 0.08)
    records = []
    for contour in _contours(mask):
        area = float(cv2.contourArea(contour))
        if area < display_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        (cx_f, cy_f), r_circle = cv2.minEnclosingCircle(contour)
        cx, cy = int(cx_f), int(cy_f)
        r_circle = float(r_circle)
        r_eq = math.sqrt(area / math.pi)
        radius = min(r_circle, r_eq)
        depth_m, depth_center, depth_box = depth_sample(
            depth, encoding, (cx, cy), frame.shape, common_cfg)
        checks = [
            _condition('area', '%.0f' % area, '>=%.0f' % min_area,
                       area >= min_area),
            _condition('depth_valid',
                       'none' if depth_m is None else '%.3fm' % depth_m,
                       '%.2f..%.2fm' % (float(common_cfg['valid_min_depth_m']),
                                        float(common_cfg['valid_max_depth_m'])),
                       depth_m is not None),
        ]
        record = {
            'color': color_name,
            'center': (cx, cy),
            'radius': float(radius),
            'radius_circle': r_circle,
            'radius_eq': float(r_eq),
            'area': area,
            'bbox': (int(bx), int(by), int(bx + bw), int(by + bh)),
            'depth_m': depth_m,
            'depth_center': depth_center,
            'depth_box': depth_box,
            'error_x': int(cx - image_center_x),
            'side': 'left' if cx < image_center_x else 'right',
            'checks': checks,
        }
        record['accepted'] = _accepted(record)
        record['note'] = '%s r=%.1f d=%s' % (
            color_name, radius,
            'none' if depth_m is None else '%.2fm' % depth_m)
        records.append(record)
    records.sort(key=lambda item: (
        item['accepted'],
        sum(check['passed'] for check in item['checks']),
        item['area']), reverse=True)
    return records, mask


def ball_scene(orange_records, blue_records, shape, cfg):
    """Mirror detect_ball_scene + compute_center_cruise_vy + left-ball avoid."""
    h, w = shape[:2]
    orange = [item for item in orange_records if item['accepted']]
    blue = [item for item in blue_records if item['accepted']]
    balls = orange + blue
    image_center_x = w // 2
    left = [item for item in balls if item['center'][0] < image_center_x]
    right = [item for item in balls if item['center'][0] >= image_center_x]
    left_ref = min(left, key=lambda b: b['depth_m']) if left else None
    right_ref = min(right, key=lambda b: b['depth_m']) if right else None
    center_error = None
    lane_mid_x = None
    if left_ref is not None and right_ref is not None:
        lane_mid_x = 0.5 * (left_ref['center'][0] + right_ref['center'][0])
        center_error = lane_mid_x - image_center_x

    if not orange:
        target = None
    elif bool(cfg['prefer_nearest_ball']):
        target = min(orange, key=lambda b: b['depth_m'])
    else:
        target = min(orange,
                     key=lambda b: b['depth_m'] + 0.002 * abs(b['error_x']))

    mode, center_vy, far_side, depth_diff = 'NO_CENTER_REF', 0.0, None, None
    if left_ref is not None and right_ref is not None:
        depth_diff = abs(float(left_ref['depth_m']) -
                         float(right_ref['depth_m']))
        if depth_diff >= float(cfg['center_depth_diff_disable_align_m']):
            far_side = ('left' if left_ref['depth_m'] > right_ref['depth_m']
                        else 'right')
            center_vy = abs(float(cfg['center_far_side_fixed_vy']))
            if far_side == 'right':
                center_vy = -center_vy
            mode = 'FAR_SIDE_BIAS'
        elif center_error is None:
            mode = 'NO_CENTER_ERR'
        elif abs(center_error) <= float(cfg['center_ok_px']):
            mode = 'CENTER_OK'
        else:
            mode = 'NORMAL_ALIGN'
            center_vy = (-abs(float(cfg['center_cruise_fixed_vy']))
                         if center_error > 0.0
                         else abs(float(cfg['center_cruise_fixed_vy'])))

    danger = None
    if bool(cfg['stage2_left_ball_avoid_enabled']):
        candidates = []
        for item in balls:
            if item['depth_m'] is None or item['center'][0] >= image_center_x:
                continue
            distance = image_center_x - float(item['center'][0])
            if distance > float(cfg['stage2_left_ball_avoid_center_px']):
                continue
            if float(item['depth_m']) > float(
                    cfg['stage2_left_ball_avoid_depth_m']):
                continue
            if float(item['radius']) < float(
                    cfg['stage2_left_ball_avoid_min_radius']):
                continue
            entry = dict(item)
            entry['distance_to_center_px'] = distance
            candidates.append(entry)
        if candidates:
            danger = min(candidates, key=lambda b: (
                float(b['depth_m']), float(b['distance_to_center_px'])))

    return {
        'img_shape': (h, w),
        'orange': orange,
        'blue': blue,
        'left_ref': left_ref,
        'right_ref': right_ref,
        'lane_mid_x': lane_mid_x,
        'center_error_px': center_error,
        'center_mode': mode,
        'center_vy': center_vy,
        'center_far_side': far_side,
        'center_depth_diff': depth_diff,
        'target': target,
        'target_aligned': (target is not None and
                           abs(target['error_x']) <=
                           float(cfg['lateral_align_px_tol'])),
        'danger_ball': danger,
    }


# ============================================================
# 黄线检测（镜像 stage2_node.detect_yellow_stop_line，保留 FAIL）
# ============================================================
def yellow_roi_box(shape, cfg):
    h, w = shape[:2]
    roi_top = max(0, min(h - 1, int(h * float(cfg['yellow_roi_top_ratio']))))
    roi_left = max(0, min(w - 1, int(w * float(cfg['yellow_roi_left_ratio']))))
    roi_right = max(roi_left + 1,
                    min(w, int(w * float(cfg['yellow_roi_right_ratio']))))
    return roi_left, roi_top, roi_right, h


def yellow_line_records(frame, cfg):
    """Mirror the yellow filter and score, keeping every rejected contour.

    ``strict`` mirrors the states that require the front horizontal line;
    ``loose`` mirrors STAGE1/STAGE3 cruise, where only the area gate applies.
    """
    roi_left, roi_top, roi_right, roi_bottom = yellow_roi_box(frame.shape, cfg)
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    roi_w = max(roi_right - roi_left, 1)
    low, high = _color_bounds(cfg, 'yellow')
    mask = _hsv_mask(roi, low, high, 5)
    min_area = float(cfg['yellow_min_contour_area'])
    min_wh = float(cfg['yellow_min_width_height_ratio'])
    min_width_ratio = float(cfg['yellow_min_width_ratio'])
    center_tolerance = float(cfg['yellow_center_tolerance_ratio'])
    display_area = max(10.0, min_area * 0.15)
    records = []
    for contour in _contours(mask):
        area = float(cv2.contourArea(contour))
        if area < display_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh <= 0:
            continue
        wh_ratio = bw / float(bh)
        width_ratio = bw / float(roi_w)
        center_offset = abs((x + bw / 2.0) - roi_w / 2.0) / float(roi_w)
        angle = _fit_line_angle(contour)
        area_check = _condition('area', '%.0f' % area, '>=%.0f' % min_area,
                                area >= min_area)
        shape_checks = [
            _condition('wh_ratio', '%.2f' % wh_ratio, '>=%.2f' % min_wh,
                       wh_ratio >= min_wh),
            _condition('width_ratio', '%.3f' % width_ratio,
                       '>=%.2f' % min_width_ratio,
                       width_ratio >= min_width_ratio),
            _condition('center_offset', '%.3f' % center_offset,
                       '<=%.2f' % center_tolerance,
                       center_offset <= center_tolerance),
        ]
        record = {
            'bbox': (roi_left + x, roi_top + y,
                     roi_left + x + bw, roi_top + y + bh),
            'bottom_y': int(roi_top + y + bh),
            'center': (int(roi_left + x + bw // 2), int(roi_top + y + bh // 2)),
            'area': area,
            'wh_ratio': float(wh_ratio),
            'width_ratio': float(width_ratio),
            'center_offset': float(center_offset),
            'angle_deg': angle,
            'score': float(y + bh),
            'checks': [area_check] + shape_checks,
            'loose_ok': area_check['passed'],
        }
        record['accepted'] = _accepted(record)
        record['note'] = 'bottom=%d angle=%.1fdeg' % (
            record['bottom_y'], angle)
        records.append(record)
    records.sort(key=lambda item: (
        item['accepted'],
        sum(check['passed'] for check in item['checks']),
        item['score']), reverse=True)
    strict = [item for item in records if item['accepted']]
    loose = [item for item in records if item['loose_ok']]
    best_strict = max(strict, key=lambda item: item['score']) if strict else None
    best_loose = max(loose, key=lambda item: item['score']) if loose else None
    return {
        'records': records,
        'best_strict': best_strict,
        'best_loose': best_loose,
        'mask': mask,
        'roi': (roi_left, roi_top, roi_right, roi_bottom),
    }


def _fit_line_angle(contour):
    """Mirror stage2_node.get_signed_yellow_line_angle_deg."""
    if contour is None or len(contour) < 2:
        return 0.0
    vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = math.degrees(math.atan2(float(vy), float(vx)))
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def yellow_align_wz(angle_deg, cfg):
    """Mirror stage2_node.compute_yellow_angle_align_wz."""
    if not bool(cfg['yellow_angle_align_enabled']) or angle_deg is None:
        return 0.0
    if abs(float(angle_deg)) <= float(cfg['yellow_angle_align_deadband_deg']):
        return 0.0
    fixed = abs(float(cfg['yellow_angle_align_fixed_wz']))
    return -fixed if float(angle_deg) > 0.0 else fixed


# ============================================================
# 可视化
# ============================================================
def fisheye_view(frame, side, orange_records, blue_records, cfg, counters,
                 encoding, colour_warning):
    """Fisheye view: candidates, entry window, hit ring, parameter panel."""
    view = frame.copy()
    h, w = view.shape[:2]
    orange_best = fisheye_best(orange_records)
    blue_best = fisheye_best(blue_records)

    # 侧撞入口居中窗口。
    cx_ratio = float(cfg['fisheye_entry_center_x_ratio'])
    cy_ratio = float(cfg['fisheye_entry_center_y_ratio'])
    x_tolerance = float(cfg['fisheye_entry_x_tolerance'])
    y_tolerance = float(cfg['fisheye_entry_y_tolerance'])
    box = (int(w * (cx_ratio - x_tolerance)), int(h * (cy_ratio - y_tolerance)),
           int(w * (cx_ratio + x_tolerance)), int(h * (cy_ratio + y_tolerance)))
    cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), WHITE_BGR, 1)
    _put_label(view, 'ENTRY WINDOW x+-%.2f y+-%.2f' % (x_tolerance, y_tolerance),
               (box[0] + 4, box[1] - 6), WHITE_BGR, 0.40)
    cv2.drawMarker(view, (int(w * cx_ratio), int(h * cy_ratio)), WHITE_BGR,
                   cv2.MARKER_CROSS, 14, 1)

    for records, base_color, tag in ((orange_records, ORANGE_BGR, 'O'),
                                     (blue_records, BLUE_BGR, 'B')):
        for index, record in enumerate(records[:6]):
            color = base_color if record['accepted'] else FAIL_BGR
            x1, y1, x2, y2 = record['bbox']
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            cv2.circle(view, record['center'],
                       int(max(2, round(record['radius']))), color, 1)
            label = '%s%d r=%.1f c=%.2f ar=%.2f' % (
                tag, index, record['radius'], record['circularity'],
                record['aspect_ratio'])
            if not record['accepted']:
                label += ' FAIL:' + ','.join(_failed_names(record, 2))
            _put_label(view, label, (_label_x(w, x1, label), max(14, y1 - 6)),
                       color, 0.40)

    hit_radius = float(cfg['fisheye_hit_radius'])
    if orange_best is not None:
        cv2.circle(view, orange_best['center'], int(hit_radius), WHITE_BGR, 1)
        _put_label(view, 'HIT r>=%.0f' % hit_radius,
                   (orange_best['center'][0] + int(hit_radius) + 4,
                    orange_best['center'][1]), WHITE_BGR, 0.40)

    entry_ok, x_ratio, y_ratio = fisheye_entry_ok(orange_best, cfg)
    approach_vx, x_error = fisheye_approach_vx(orange_best, cfg, side)
    sign = float(cfg['left_fisheye_x_to_vx_sign'] if side == 'left'
                 else cfg['right_fisheye_x_to_vx_sign'])
    entry_frames = int(cfg['fisheye_entry_confirm_frames'])
    hit_frames = int(cfg['fisheye_hit_radius_confirm_frames'])

    lines = [
        ('HSV   orange %s | blue %s' % (
            _hsv_text(cfg, 'orange'), _hsv_text(cfg, 'blue')), GREY_BGR),
        ('SHAPE area>=%.0f(O)/%.0f(B) circ>=%.2f aspect %.2f..%.2f '
         'circle_fill>=%.2f bbox_fill>=%.2f' % (
             float(cfg['fisheye_orange_min_contour_area']),
             float(cfg['fisheye_blue_min_contour_area']),
             float(cfg['fisheye_min_circularity']),
             float(cfg['fisheye_min_aspect_ratio']),
             float(cfg['fisheye_max_aspect_ratio']),
             float(cfg['fisheye_min_circle_fill_ratio']),
             float(cfg['fisheye_min_bbox_fill_ratio'])), GREY_BGR),
        ('MORPH open+close k=%dx%d   radius=min(minEnclosingCircle, '
         'sqrt(area/pi))' % (
             _odd_kernel(cfg['fisheye_morph_kernel_size']),
             _odd_kernel(cfg['fisheye_morph_kernel_size'])), GREY_BGR),
        ('ENTRY center=(%.2f,%.2f) tol=(%.2f,%.2f) confirm=%d -> %s '
         'x=%s y=%s streak=%d' % (
             cx_ratio, cy_ratio, x_tolerance, y_tolerance, entry_frames,
             'YES' if entry_ok else 'NO',
             '-' if x_ratio is None else '%.3f' % x_ratio,
             '-' if y_ratio is None else '%.3f' % y_ratio,
             counters.get('entry', 0)),
         PASS_BGR if entry_ok else GREY_BGR),
        ('HIT   radius>=%.1f confirm=%d -> r=%s streak=%d  hit_vy=%.2f' % (
            hit_radius, hit_frames,
            '-' if orange_best is None else '%.1f' % orange_best['radius'],
            counters.get('hit', 0), float(cfg['fisheye_hit_vy'])),
         PASS_BGR if (orange_best is not None and
                      orange_best['radius'] >= hit_radius) else GREY_BGR),
        ('APPR  target_x=%.2f dead=%.2f k=%.2f vx_max=%.2f sign=%+.1f '
         '-> x_err=%s vx=%+.3f' % (
             float(cfg['fisheye_approach_target_x_ratio']),
             float(cfg['fisheye_approach_x_deadband_ratio']),
             float(cfg['fisheye_approach_vx_k']),
             float(cfg['fisheye_approach_vx_max']), sign,
             '-' if x_error is None else '%+.3f' % x_error, approach_vx),
         GREY_BGR),
        ('      approach_vy=%.2f lost_frames=%d' % (
            float(cfg['fisheye_approach_vy']),
            int(cfg['fisheye_approach_lost_frames'])), GREY_BGR),
        ('BLUE  best %s   (preview only: stage2_node hits orange only)' % (
            'none' if blue_best is None else 'r=%.1f at %s' % (
                blue_best['radius'], blue_best['center'])), GREY_BGR),
        ('SRC   encoding=%s%s' % (encoding or 'unknown', colour_warning),
         FAIL_BGR if colour_warning else GREY_BGR),
    ]
    view = _append_status_panel(
        view, '%s FISHEYE - orange = side-hit target - live values vs rules'
        % side.upper(), lines, 44 + 17 * len(lines))
    return _append_condition_row(view, [
        ('ORANGE BEST', orange_records[0] if orange_records else None),
        ('ORANGE 2ND', orange_records[1] if len(orange_records) > 1 else None),
        ('BLUE BEST', blue_records[0] if blue_records else None),
    ])


def rgb_ball_view(frame, scene, orange_records, blue_records, cfg, depth_shape):
    """RGB view: every candidate, lane centre, target ball, avoid gate."""
    view = frame.copy()
    h, w = view.shape[:2]
    image_center_x = w // 2
    cv2.line(view, (image_center_x, 0), (image_center_x, h - 1), WHITE_BGR, 1)

    for records, base_color, tag in ((orange_records, ORANGE_BGR, 'O'),
                                     (blue_records, BLUE_BGR, 'B')):
        for index, record in enumerate(records[:8]):
            color = base_color if record['accepted'] else FAIL_BGR
            cx, cy = record['center']
            radius = int(max(2, round(record['radius'])))
            cv2.circle(view, (cx, cy), radius, color, 2)
            cv2.circle(view, (cx, cy), 3, color, -1)
            depth_m = record['depth_m']
            label = '%s%d r=%.1f d=%s ex=%d' % (
                tag, index, record['radius'],
                'none' if depth_m is None else '%.2fm' % depth_m,
                record['error_x'])
            if not record['accepted']:
                label += ' FAIL:' + ','.join(_failed_names(record, 2))
            _put_label(view, label, (_label_x(w, cx - 48, label),
                                     max(14, cy - radius - 7)), color, 0.40)
            box = record['depth_box']
            if box is not None and depth_shape is not None:
                dh, dw = depth_shape[:2]
                x1, y1, x2, y2 = box
                cv2.rectangle(view,
                              (int(x1 * w / max(dw, 1)), int(y1 * h / max(dh, 1))),
                              (int(x2 * w / max(dw, 1)), int(y2 * h / max(dh, 1))),
                              color, 1)

    for name, ref in (('LEFT_REF', scene['left_ref']),
                      ('RIGHT_REF', scene['right_ref'])):
        if ref is None:
            continue
        cx, cy = ref['center']
        cv2.circle(view, (cx, cy), 9, (255, 255, 0), 2)
        _put_label(view, name, (cx + 10, cy), (255, 255, 0), 0.42)

    if scene['lane_mid_x'] is not None:
        mid = int(scene['lane_mid_x'])
        cv2.line(view, (mid, 0), (mid, h - 1), PASS_BGR, 2)
        text = 'lane_mid=%d err=%.1f' % (mid, scene['center_error_px'])
        _put_label(view, text, (_label_x(w, mid + 6, text, 0.44), 24),
                   PASS_BGR, 0.44)

    target = scene['target']
    if target is not None:
        cx, cy = target['center']
        radius = int(max(8, round(target['radius'])))
        cv2.circle(view, (cx, cy), radius + 5, FAIL_BGR, 2)
        text = 'TARGET %s d=%.2fm ex=%d aligned=%s' % (
            target['color'], target['depth_m'], target['error_x'],
            'YES' if scene['target_aligned'] else 'NO')
        _put_label(view, text, (_label_x(w, cx - 70, text, 0.44),
                                min(h - 6, cy + radius + 20)), FAIL_BGR, 0.44)

    danger = scene['danger_ball']
    if danger is not None:
        cx, cy = danger['center']
        radius = int(max(8, round(danger['radius'])))
        cv2.circle(view, (cx, cy), radius + 11, (0, 165, 255), 2)
        text = 'S2 AVOID %s dist=%.0fpx' % (
            danger['color'], danger['distance_to_center_px'])
        _put_label(view, text, (_label_x(w, cx - 70, text, 0.44),
                                min(h - 6, cy + radius + 40)), (0, 165, 255),
                   0.44)

    avoid_center = float(cfg['stage2_left_ball_avoid_center_px'])
    cv2.line(view, (int(image_center_x - avoid_center), 0),
             (int(image_center_x - avoid_center), h - 1), (0, 165, 255), 1)
    _put_label(view, 'avoid gate x<=%.0fpx' % avoid_center,
               (max(3, int(image_center_x - avoid_center) + 4), h - 8),
               (0, 165, 255), 0.40)

    depth_diff = scene['center_depth_diff']
    error_text = ('-' if scene['center_error_px'] is None
                  else '%.1fpx' % scene['center_error_px'])
    lines = [
        ('HSV   orange %s area>=%.0f' % (
            _hsv_text(cfg, 'orange'),
            float(cfg['orange_min_contour_area'])), GREY_BGR),
        ('      blue   %s area>=%.0f' % (
            _hsv_text(cfg, 'blue'),
            float(cfg['blue_min_contour_area'])), GREY_BGR),
        ('MORPH open+close 5x5 (fixed)  radius=min(minEnclosingCircle, '
         'sqrt(area/pi))', GREY_BGR),
        ('DEPTH +-%dpx window, valid %.2f..%.2fm, 20th pct; '
         'no depth -> candidate dropped' % (
             int(cfg['depth_search_half']), float(cfg['valid_min_depth_m']),
             float(cfg['valid_max_depth_m'])), GREY_BGR),
        ('CAND  orange=%d/%d blue=%d/%d (passed/visible)' % (
            len(scene['orange']), len(orange_records),
            len(scene['blue']), len(blue_records)), GREY_BGR),
        ('LANE  center_ok_px=%.1f fixed_vy=%.2f depth_diff_off=%.2f '
         'far_vy=%.2f' % (
             float(cfg['center_ok_px']), float(cfg['center_cruise_fixed_vy']),
             float(cfg['center_depth_diff_disable_align_m']),
             float(cfg['center_far_side_fixed_vy'])), GREY_BGR),
        ('      -> mode=%s vy=%+.3f err=%s depth_diff=%s' % (
            scene['center_mode'], scene['center_vy'], error_text,
            '-' if depth_diff is None else '%.3f' % depth_diff),
         PASS_BGR if scene['center_mode'] == 'CENTER_OK' else GREY_BGR),
        ('TGT   lateral_align_px_tol=%.1f prefer_nearest=%s -> %s' % (
            float(cfg['lateral_align_px_tol']),
            bool(cfg['prefer_nearest_ball']),
            'none' if target is None else '%s d=%.2fm ex=%d' % (
                target['color'], target['depth_m'], target['error_x'])),
         GREY_BGR),
        ('AVOID on=%s center<=%.0fpx depth<=%.2fm radius>=%.1f confirm=%d '
         'vy=%.2f' % (
             bool(cfg['stage2_left_ball_avoid_enabled']), avoid_center,
             float(cfg['stage2_left_ball_avoid_depth_m']),
             float(cfg['stage2_left_ball_avoid_min_radius']),
             int(cfg['stage2_left_ball_avoid_confirm_frames']),
             float(cfg['stage2_left_ball_avoid_vy'])), GREY_BGR),
        ('      -> %s' % (
            'none' if danger is None else '%s d=%.2fm dist=%.0fpx' % (
                danger['color'], danger['depth_m'],
                danger['distance_to_center_px'])),
         (0, 165, 255) if danger is not None else GREY_BGR),
    ]
    view = _append_status_panel(
        view, 'RGB BALLS - detect_ball_scene mirror - live values vs rules',
        lines, 44 + 17 * len(lines))
    return _append_condition_row(view, [
        ('ORANGE BEST', orange_records[0] if orange_records else None),
        ('ORANGE 2ND', orange_records[1] if len(orange_records) > 1 else None),
        ('BLUE BEST', blue_records[0] if blue_records else None),
    ])


def rgb_yellow_view(frame, yellow, cfg):
    """Yellow-line view: ROI, thresholds, strict/loose selection, angle."""
    view = frame.copy()
    h, w = view.shape[:2]
    roi_left, roi_top, roi_right, roi_bottom = yellow['roi']
    cv2.rectangle(view, (roi_left, roi_top), (roi_right - 1, roi_bottom - 1),
                  YELLOW_BGR, 1)
    _put_label(view, 'yellow ROI top=%.2f x=%.2f..%.2f' % (
        float(cfg['yellow_roi_top_ratio']),
        float(cfg['yellow_roi_left_ratio']),
        float(cfg['yellow_roi_right_ratio'])),
        (roi_left + 4, max(14, roi_top - 6)), YELLOW_BGR, 0.42)

    trigger_rules = [
        ('stage1', float(cfg['yellow_stop_line_y_ratio_stage1'])),
        ('stage2', float(cfg['yellow_stop_line_y_ratio_stage2'])),
        ('stage3', float(cfg['yellow_stop_line_y_ratio_stage3'])),
        ('scan', float(cfg['yellow_ratio_scan'])),
        ('final', float(cfg['yellow_ratio_final'])),
    ]
    slowdown_rules = [
        ('slow1', float(cfg['yellow_slowdown_ratio_stage1'])),
        ('slow2', float(cfg['yellow_slowdown_ratio_stage2'])),
        ('slow3', float(cfg['yellow_slowdown_ratio_stage3'])),
        ('slow_scan', float(cfg['yellow_slowdown_ratio_scan'])),
        ('slow_final', float(cfg['yellow_slowdown_ratio_final'])),
    ]
    drawn = {}
    for name, ratio in slowdown_rules:
        y = max(0, min(h - 1, int(h * ratio)))
        cv2.line(view, (0, y), (w - 1, y), (120, 190, 255), 1)
        drawn.setdefault(y, []).append(name)
    for name, ratio in trigger_rules:
        y = max(0, min(h - 1, int(h * ratio)))
        cv2.line(view, (0, y), (w - 1, y), (0, 180, 255), 2)
        drawn.setdefault(y, []).append(name)
    for y, names in drawn.items():
        text = '%s y=%d' % ('/'.join(names), y)
        _put_label(view, text, (_label_x(w, w - 240, text), max(14, y - 5)),
                   (0, 180, 255), 0.40)

    for index, record in enumerate(yellow['records'][:6]):
        color = YELLOW_BGR if record['accepted'] else FAIL_BGR
        x1, y1, x2, y2 = record['bbox']
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 1)
        label = 'Y%d wh=%.1f wr=%.2f off=%.2f' % (
            index, record['wh_ratio'], record['width_ratio'],
            record['center_offset'])
        if not record['accepted']:
            label += ' FAIL:' + ','.join(_failed_names(record, 2))
        _put_label(view, label, (_label_x(w, x1, label), max(14, y1 - 5)),
                   color, 0.40)

    best_strict = yellow['best_strict']
    best_loose = yellow['best_loose']
    for record, color, name in ((best_loose, (255, 160, 0), 'LOOSE'),
                                (best_strict, YELLOW_BGR, 'STRICT')):
        if record is None:
            continue
        x1, y1, x2, y2 = record['bbox']
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.line(view, (0, record['bottom_y']), (w - 1, record['bottom_y']),
                 color, 2)
        _put_label(view, '%s bottom=%d angle=%.1fdeg' % (
            name, record['bottom_y'], record['angle_deg']),
            (6, max(14, record['bottom_y'] - 6)), color, 0.44)
        cx, cy = record['center']
        length = 80
        radians = math.radians(record['angle_deg'])
        dx = int(math.cos(radians) * length)
        dy = int(math.sin(radians) * length)
        cv2.line(view, (cx - dx, cy - dy), (cx + dx, cy + dy), FAIL_BGR, 2)

    strict_angle = best_strict['angle_deg'] if best_strict else None
    loose_angle = best_loose['angle_deg'] if best_loose else None
    angle = strict_angle if strict_angle is not None else loose_angle
    wz = yellow_align_wz(angle, cfg)
    reached = []
    for name, ratio in trigger_rules:
        record = best_loose if name in ('stage1', 'stage3', 'scan',
                                        'final') else best_strict
        if record is not None and record['bottom_y'] >= int(h * ratio):
            reached.append(name)

    lines = [
        ('HSV    %s area>=%.0f  morph open+close 5x5' % (
            _hsv_text(cfg, 'yellow'),
            float(cfg['yellow_min_contour_area'])), GREY_BGR),
        ('ROI    top=%.2f x=%.2f..%.2f of the full frame' % (
            float(cfg['yellow_roi_top_ratio']),
            float(cfg['yellow_roi_left_ratio']),
            float(cfg['yellow_roi_right_ratio'])), GREY_BGR),
        ('STRICT front-line gates: wh_ratio>=%.2f width_ratio>=%.2f '
         'center_offset<=%.2f' % (
             float(cfg['yellow_min_width_height_ratio']),
             float(cfg['yellow_min_width_ratio']),
             float(cfg['yellow_center_tolerance_ratio'])), GREY_BGR),
        ('       used by STAGE2_CRUISE_YELLOW_ONLY and the timed states',
         GREY_BGR),
        ('LOOSE  area only, used by %s' % ' / '.join(
            _short_state(state) for state in YELLOW_LOOSE_STATES), GREY_BGR),
        ('PICK   score=y+bh (lowest line wins) -> strict=%s loose=%s' % (
            'none' if best_strict is None
            else 'bottom=%d' % best_strict['bottom_y'],
            'none' if best_loose is None
            else 'bottom=%d' % best_loose['bottom_y']),
         PASS_BGR if best_strict is not None else GREY_BGR),
        ('TRIG   ' + ' '.join('%s=%.2f' % item for item in trigger_rules) +
         ' confirm=%d' % int(cfg['yellow_stop_confirm_count']), GREY_BGR),
        ('SLOW   ' + ' '.join('%s=%.2f' % item for item in slowdown_rules),
         GREY_BGR),
        ('REACH  %s' % (', '.join(reached) if reached else 'none'),
         PASS_BGR if reached else GREY_BGR),
        ('ANGLE  on=%s dead=%.2fdeg fixed_wz=%.2f -> angle=%s wz=%+.3f' % (
            bool(cfg['yellow_angle_align_enabled']),
            float(cfg['yellow_angle_align_deadband_deg']),
            float(cfg['yellow_angle_align_fixed_wz']),
            '-' if angle is None else '%+.2f' % angle, wz), GREY_BGR),
        ('       yellow_max_tilt_deg=%.1f is read by stage2_node but never '
         'filters' % float(cfg['yellow_max_tilt_deg']), GREY_BGR),
    ]
    view = _append_status_panel(
        view, 'RGB YELLOW LINE - live values vs rules', lines,
        44 + 17 * len(lines))
    return _append_condition_row(view, [
        ('YELLOW BEST', yellow['records'][0] if yellow['records'] else None),
        ('YELLOW 2ND',
         yellow['records'][1] if len(yellow['records']) > 1 else None),
    ])


def depth_view(depth, encoding, shape):
    if depth is None or encoding not in ('16UC1', '32FC1'):
        return _blank_view((shape[1], shape[0]),
                           'NO DEPTH FRAME (%s)' % (encoding or '-'))
    if encoding == '16UC1':
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)
    depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
    valid = depth_m > 0.0
    clipped = np.clip(depth_m, 0.2, 3.0)
    normalized = ((clipped - 0.2) / 2.8 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    _put_label(colored, 'DEPTH 0.2m..3.0m  encoding=%s' % encoding,
               (10, 22), WHITE_BGR, 0.50)
    _put_label(colored, 'ball depth = 20th pct of a +-N px window; RGB and '
                        'depth are not registered on the robot,',
               (10, 44), (0, 165, 255), 0.42)
    _put_label(colored, 'the sample point is only scaled by resolution ratio',
               (10, 64), (0, 165, 255), 0.42)
    return colored


# ============================================================
# 预览状态与节点
# ============================================================
class PreviewState(object):
    """Latest-frame buffers plus rendered JPEG frames for the web page."""

    CHANNELS = ('rgb', 'depth', 'fisheye_left', 'fisheye_right')

    def __init__(self, stale_after):
        self.lock = threading.Lock()
        self.stale_after = float(stale_after)
        self.messages = {name: None for name in self.CHANNELS}
        self.sequences = {name: 0 for name in self.CHANNELS}
        self.received = {name: 0 for name in self.CHANNELS}
        self.last_rx = {name: 0.0 for name in self.CHANNELS}
        self.publishers = {name: 0 for name in self.CHANNELS}
        self.shapes = {name: None for name in self.CHANNELS}
        self.processed = 0
        self.dropped = 0
        self.frames = {key: None for key in STREAMS}
        self.frame_seq = 0
        self.processing_started = 0.0
        self.processing_step = 'idle'
        self.last_process = 0.0
        self.last_process_duration = None
        self.process_times = collections.deque(maxlen=60)
        self.warning = ''
        self.last_error = ''
        self.summary = {}

    def push(self, name, msg):
        with self.lock:
            if name in ('rgb', 'fisheye_left', 'fisheye_right'):
                if self.sequences[name] > self.processed:
                    self.dropped += 1
            self.messages[name] = msg
            self.sequences[name] += 1
            self.received[name] += 1
            self.last_rx[name] = time.monotonic()
            self.shapes[name] = '%dx%d %s' % (
                msg.width, msg.height, msg.encoding)

    def make_callback(self, name):
        def callback(msg):
            self.push(name, msg)
        return callback

    def claim(self):
        """Return the latest frame set when any image channel advanced."""
        with self.lock:
            newest = max(self.sequences[name] for name in
                         ('rgb', 'fisheye_left', 'fisheye_right'))
            if newest <= self.processed:
                return None
            self.processed = newest
            return dict(self.messages)

    def begin_processing(self):
        with self.lock:
            self.processing_started = time.monotonic()
            self.processing_step = 'decode'

    def set_processing_step(self, step):
        with self.lock:
            self.processing_step = str(step)

    def commit(self, frames, summary, warning):
        now = time.monotonic()
        with self.lock:
            self.frames = dict(frames)
            self.frame_seq += 1
            self.summary = dict(summary)
            self.warning = str(warning or '')
            self.last_process = now
            if self.processing_started:
                self.last_process_duration = now - self.processing_started
            self.processing_started = 0.0
            self.processing_step = 'idle'
            self.process_times.append(now)
            self.last_error = ''

    def error(self, error):
        with self.lock:
            now = time.monotonic()
            if self.processing_started:
                self.last_process_duration = now - self.processing_started
            self.processing_started = 0.0
            self.processing_step = 'idle'
            self.last_error = str(error)
            self.warning = '处理失败：%s' % error

    def processing_snapshot(self):
        with self.lock:
            age = 0.0
            if self.processing_started:
                age = max(0.0, time.monotonic() - self.processing_started)
            return age, self.processing_step

    def set_publishers(self, counts):
        with self.lock:
            self.publishers.update(counts)

    def frame(self, key, after):
        with self.lock:
            if key not in self.frames or self.frames[key] is None:
                return None
            if after is not None and after >= self.frame_seq:
                return self.frame_seq, None, 0.0
            age = max(0.0, time.monotonic() - self.last_process)
            return self.frame_seq, self.frames[key], age

    def _age(self, name, now):
        return (max(0.0, now - self.last_rx[name])
                if self.last_rx[name] else None)

    def status(self, node):
        now = time.monotonic()
        with self.lock:
            ages = {name: self._age(name, now) for name in self.CHANNELS}
            process_age = (max(0.0, now - self.processing_started)
                           if self.processing_started else None)
            warning = self.warning
            fresh = [name for name in ('rgb', 'fisheye_left', 'fisheye_right')
                     if ages[name] is not None
                     and ages[name] <= self.stale_after]
            if self.last_error:
                health = 'error'
            elif (process_age is not None
                  and process_age > node.args.processing_timeout):
                health = 'error'
                warning = ('处理线程停在 %s 已 %.1f 秒，请查看终端 '
                           'PREVIEW_STALL 日志。') % (
                               self.processing_step, process_age)
            elif not fresh:
                health = 'waiting'
            else:
                health = 'live'
            fps = None
            if len(self.process_times) >= 2:
                elapsed = self.process_times[-1] - self.process_times[0]
                if elapsed > 0:
                    fps = (len(self.process_times) - 1) / elapsed

            def age_text(name):
                value = ages[name]
                return '-' if value is None else '%.2fs' % value

            summary = self.summary
            camera_text = '\n'.join([
                '左鱼眼: %s' % node.topics['fisheye_left'],
                '右鱼眼: %s' % node.topics['fisheye_right'],
                'RGB   : %s' % node.topics['rgb'],
                'Depth : %s' % node.topics['depth'],
                '尺寸  : L %s | R %s' % (
                    self.shapes['fisheye_left'] or '-',
                    self.shapes['fisheye_right'] or '-'),
                '        RGB %s | D %s' % (
                    self.shapes['rgb'] or '-', self.shapes['depth'] or '-'),
                '帧数  : L %d / R %d / RGB %d / D %d' % (
                    self.received['fisheye_left'],
                    self.received['fisheye_right'],
                    self.received['rgb'], self.received['depth']),
                '延迟  : L %s / R %s / RGB %s / D %s' % (
                    age_text('fisheye_left'), age_text('fisheye_right'),
                    age_text('rgb'), age_text('depth')),
                '发布者: L %d / R %d / RGB %d / D %d' % (
                    self.publishers['fisheye_left'],
                    self.publishers['fisheye_right'],
                    self.publishers['rgb'], self.publishers['depth']),
                'platform=%s  处理=%s FPS  单帧=%s  丢弃旧帧=%d' % (
                    node.platform,
                    '-' if fps is None else '%.2f' % fps,
                    '-' if self.last_process_duration is None
                    else '%.3fs' % self.last_process_duration,
                    self.dropped),
            ])
            return {
                'health': health,
                'warning': warning,
                'summary': 'L %d 帧 · R %d 帧 · RGB %d 帧 · %s' % (
                    self.received['fisheye_left'],
                    self.received['fisheye_right'],
                    self.received['rgb'], age_text('rgb')),
                'camera_text': camera_text,
                'fisheye_text': summary.get('fisheye_text', '等待数据'),
                'balls_text': summary.get('balls_text', '等待数据'),
                'yellow_text': summary.get('yellow_text', '等待数据'),
            }


class Stage2VisionPreviewNode(Node):
    """Read-only Stage-2 detection preview served over HTTP."""

    def __init__(self, args):
        Node.__init__(self, 'stage2_vision_preview')
        self.args = args
        self.bridge = CvBridge()
        cv2.setNumThreads(max(1, int(args.opencv_threads)))
        self.declare_parameter('platform', args.platform)
        self.platform = str(self.get_parameter('platform').value).strip().lower()
        if self.platform not in ('sim', 'real'):
            raise ValueError("platform must be 'sim' or 'real'")
        self.declare_parameter('rgb_topic', args.rgb_topic or '')
        self.declare_parameter('depth_topic', args.depth_topic or '')
        self.declare_parameter('fisheye_left_topic', args.fisheye_left or '')
        self.declare_parameter('fisheye_right_topic', args.fisheye_right or '')

        self.cfg = collections.OrderedDict()
        for defaults in (ORANGE_DEFAULTS, BLUE_DEFAULTS, FISHEYE_DEFAULTS,
                         RGB_BALL_DEFAULTS, YELLOW_DEFAULTS):
            _declare(self, defaults)
            self.cfg.update(_read(self, defaults))

        self.state = PreviewState(args.stale_after)
        self.topics = {name: '' for name in PreviewState.CHANNELS}
        self.counters = {
            'left': {'entry': 0, 'hit': 0},
            'right': {'entry': 0, 'hit': 0},
        }
        self.avoid_counter = 0
        self._subscriptions = []
        self._publisher_timer = self.create_timer(1.0, self._update_publishers)
        self._watchdog_timer = self.create_timer(1.0, self._check_worker)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop,
                                        name='stage2-preview-worker')
        self._worker.daemon = True

    def attach(self, topics):
        self.topics.update(topics)
        qos = _image_qos()
        for name in PreviewState.CHANNELS:
            self._subscriptions.append(self.create_subscription(
                Image, self.topics[name], self.state.make_callback(name), qos))
            self.get_logger().info('%s topic: %s' % (name, self.topics[name]))
        self._worker.start()
        self._update_publishers()

    def close(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def _update_publishers(self):
        counts = {}
        for name in PreviewState.CHANNELS:
            if not self.topics[name]:
                counts[name] = 0
                continue
            try:
                counts[name] = self.count_publishers(self.topics[name])
            except Exception:
                counts[name] = 0
        self.state.set_publishers(counts)

    def _check_worker(self):
        age, step = self.state.processing_snapshot()
        if age > self.args.processing_timeout:
            self.get_logger().error(
                '[PREVIEW_STALL] processing stuck at %s for %.1fs' % (step, age),
                throttle_duration_sec=2.0)

    def _worker_loop(self):
        period = 1.0 / max(float(self.args.max_fps), 0.1)
        next_time = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_time:
                self._stop_event.wait(min(0.03, next_time - now))
                continue
            item = self.state.claim()
            if item is None:
                self._stop_event.wait(0.02)
                continue
            next_time = time.monotonic() + period
            self.state.begin_processing()
            try:
                self._process(item)
            except Exception as error:
                self.state.error(error)
                self.get_logger().warning('preview processing failed: %s'
                                          % error)

    def _decode(self, msg, encoding='bgr8'):
        if msg is None:
            return None
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

    def _update_fisheye_counters(self, side, orange_best):
        counters = self.counters[side]
        entry_ok, _, _ = fisheye_entry_ok(orange_best, self.cfg)
        counters['entry'] = counters['entry'] + 1 if entry_ok else 0
        hit_ok = (orange_best is not None and orange_best['radius']
                  >= float(self.cfg['fisheye_hit_radius']))
        counters['hit'] = counters['hit'] + 1 if hit_ok else 0
        return entry_ok, hit_ok

    def _fisheye_summary(self, side, orange_best, blue_best, entry_ok, hit_ok):
        counters = self.counters[side]
        name = '左' if side == 'left' else '右'
        radius = '-' if orange_best is None else '%.1f' % orange_best['radius']
        center = '-' if orange_best is None else str(orange_best['center'])
        blue_text = ('-' if blue_best is None
                     else 'r=%.1f %s' % (blue_best['radius'],
                                         blue_best['center']))
        vx, _ = fisheye_approach_vx(orange_best, self.cfg, side)
        return '\n'.join([
            '%s鱼眼 橙球 r=%s center=%s' % (name, radius, center),
            '  入口居中=%s 连续=%d/%d' % (
                'YES' if entry_ok else 'NO', counters['entry'],
                int(self.cfg['fisheye_entry_confirm_frames'])),
            '  撞击半径=%s 连续=%d/%d (阈值 %.1f)' % (
                'YES' if hit_ok else 'NO', counters['hit'],
                int(self.cfg['fisheye_hit_radius_confirm_frames']),
                float(self.cfg['fisheye_hit_radius'])),
            '  接近 vx=%+.3f' % vx,
            '  蓝球(仅预览) %s' % blue_text,
        ])

    def _process(self, item):
        self.state.set_processing_step('decode')
        left = self._decode(item['fisheye_left'])
        right = self._decode(item['fisheye_right'])
        rgb = self._decode(item['rgb'])
        depth_msg = item['depth']
        depth = self._decode(depth_msg, 'passthrough')
        depth_encoding = depth_msg.encoding if depth_msg is not None else ''
        frames = {}
        warnings = []
        summary = {}
        size = (640, 480)
        if rgb is not None:
            size = (rgb.shape[1], rgb.shape[0])
        elif left is not None:
            size = (left.shape[1], left.shape[0])

        # ---------------- 鱼眼 ----------------
        self.state.set_processing_step('fisheye')
        fisheye_masks = []
        fisheye_summaries = []
        for side, frame, key, msg in (
                ('left', left, 'fisheye_left', item['fisheye_left']),
                ('right', right, 'fisheye_right', item['fisheye_right'])):
            if frame is None:
                frames[key] = _encode_jpeg(
                    _blank_view(size, 'WAITING FOR %s FISHEYE'
                                % side.upper()), self.args.jpeg_quality)
                fisheye_masks.append(np.zeros(size[::-1], np.uint8))
                fisheye_masks.append(np.zeros(size[::-1], np.uint8))
                self.counters[side] = {'entry': 0, 'hit': 0}
                fisheye_summaries.append('%s鱼眼 无图像' % (
                    '左' if side == 'left' else '右'))
                continue
            orange_records, orange_mask = fisheye_ball_records(
                frame, self.cfg, self.cfg, 'orange')
            blue_records, blue_mask = fisheye_ball_records(
                frame, self.cfg, self.cfg, 'blue')
            orange_best = fisheye_best(orange_records)
            blue_best = fisheye_best(blue_records)
            entry_ok, hit_ok = self._update_fisheye_counters(side, orange_best)
            saturation = float(cv2.cvtColor(
                frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            colour_warning = ''
            if saturation < 4.0:
                colour_warning = ('  mean S=%.1f: nearly greyscale, HSV '
                                  'colour detection cannot work' % saturation)
                warnings.append('%s鱼眼画面接近灰度（平均饱和度 %.1f），'
                                'HSV 橙/蓝球检测不会有结果。'
                                % ('左' if side == 'left' else '右',
                                   saturation))
            encoding = msg.encoding if msg is not None else ''
            frames[key] = _encode_jpeg(
                fisheye_view(frame, side, orange_records, blue_records,
                             self.cfg, self.counters[side], encoding,
                             colour_warning),
                self.args.jpeg_quality)
            fisheye_masks.append(orange_mask)
            fisheye_masks.append(blue_mask)
            fisheye_summaries.append(self._fisheye_summary(
                side, orange_best, blue_best, entry_ok, hit_ok))
        summary['fisheye_text'] = '\n'.join(fisheye_summaries)

        tile = (max(1, size[0] // 2), max(1, size[1] // 2))
        labels = ['L ORANGE', 'L BLUE', 'R ORANGE', 'R BLUE']
        frames['fisheye_masks'] = _encode_jpeg(
            _mosaic([_mask_tile(mask, label, tile)
                     for mask, label in zip(fisheye_masks, labels)], 2),
            self.args.jpeg_quality)

        # ---------------- RGB ----------------
        self.state.set_processing_step('rgb')
        if rgb is None:
            blank = _blank_view(size, 'WAITING FOR RGB CAMERA')
            for key in ('rgb_balls', 'rgb_yellow', 'rgb_masks'):
                frames[key] = _encode_jpeg(blank, self.args.jpeg_quality)
            summary['balls_text'] = 'RGB 无图像'
            summary['yellow_text'] = 'RGB 无图像'
        else:
            if depth is None:
                warnings.append('尚未收到深度图：RGB 球候选全部会被 '
                                'depth_valid 条件丢弃。')
            elif depth_encoding not in ('16UC1', '32FC1'):
                warnings.append('深度编码 %s 不是 16UC1/32FC1，'
                                'stage2_node 会直接返回空深度。'
                                % depth_encoding)
            orange_records, orange_mask = rgb_ball_records(
                rgb, depth, depth_encoding, self.cfg, self.cfg, 'orange')
            blue_records, blue_mask = rgb_ball_records(
                rgb, depth, depth_encoding, self.cfg, self.cfg, 'blue')
            scene = ball_scene(orange_records, blue_records, rgb.shape,
                               self.cfg)
            if scene['danger_ball'] is not None:
                self.avoid_counter += 1
            else:
                self.avoid_counter = 0
            frames['rgb_balls'] = _encode_jpeg(
                rgb_ball_view(rgb, scene, orange_records, blue_records,
                              self.cfg,
                              None if depth is None else depth.shape),
                self.args.jpeg_quality)

            self.state.set_processing_step('yellow')
            yellow = yellow_line_records(rgb, self.cfg)
            frames['rgb_yellow'] = _encode_jpeg(
                rgb_yellow_view(rgb, yellow, self.cfg), self.args.jpeg_quality)

            yellow_full = np.zeros(rgb.shape[:2], np.uint8)
            roi_left, roi_top, roi_right, roi_bottom = yellow['roi']
            yellow_full[roi_top:roi_bottom, roi_left:roi_right] = yellow['mask']
            frames['rgb_masks'] = _encode_jpeg(
                _mosaic([
                    _mask_tile(orange_mask, 'RGB ORANGE', tile),
                    _mask_tile(blue_mask, 'RGB BLUE', tile),
                    _mask_tile(yellow_full, 'RGB YELLOW ROI', tile),
                    _mask_tile(np.zeros(rgb.shape[:2], np.uint8), '', tile),
                ], 2), self.args.jpeg_quality)
            summary['balls_text'] = self._balls_summary(
                scene, orange_records, blue_records)
            summary['yellow_text'] = self._yellow_summary(yellow, rgb.shape)

        self.state.set_processing_step('depth')
        frames['depth'] = _encode_jpeg(
            depth_view(depth, depth_encoding, (size[1], size[0])),
            self.args.jpeg_quality)
        self.state.commit(frames, summary, ' '.join(warnings))

    def _balls_summary(self, scene, orange_records, blue_records):
        target = scene['target']
        danger = scene['danger_ball']
        return '\n'.join([
            '橙球 %d/%d 通过, 蓝球 %d/%d 通过' % (
                len(scene['orange']), len(orange_records),
                len(scene['blue']), len(blue_records)),
            '左参考 %s, 右参考 %s' % (
                '-' if scene['left_ref'] is None
                else '%.2fm' % scene['left_ref']['depth_m'],
                '-' if scene['right_ref'] is None
                else '%.2fm' % scene['right_ref']['depth_m']),
            '中线 err=%s mode=%s vy=%+.3f' % (
                '-' if scene['center_error_px'] is None
                else '%.1fpx' % scene['center_error_px'],
                scene['center_mode'], scene['center_vy']),
            '目标球 %s' % (
                '-' if target is None else '%s d=%.2fm ex=%d aligned=%s' % (
                    target['color'], target['depth_m'], target['error_x'],
                    'YES' if scene['target_aligned'] else 'NO')),
            '左球避让 %s 连续=%d/%d' % (
                '-' if danger is None else '%s d=%.2fm' % (
                    danger['color'], danger['depth_m']),
                self.avoid_counter,
                int(self.cfg['stage2_left_ball_avoid_confirm_frames'])),
        ])

    def _yellow_summary(self, yellow, shape):
        h = shape[0]
        strict = yellow['best_strict']
        loose = yellow['best_loose']
        angle = None
        if strict is not None:
            angle = strict['angle_deg']
        elif loose is not None:
            angle = loose['angle_deg']
        return '\n'.join([
            '候选 %d 条, strict 通过 %d 条' % (
                len(yellow['records']),
                sum(1 for item in yellow['records'] if item['accepted'])),
            'strict bottom=%s' % (
                '-' if strict is None else '%d / %d..%d' % (
                    strict['bottom_y'],
                    int(h * float(self.cfg['yellow_ratio_scan'])),
                    int(h * float(self.cfg['yellow_ratio_final'])))),
            'loose  bottom=%s' % (
                '-' if loose is None else str(loose['bottom_y'])),
            '角度 %s -> wz=%+.3f (deadband %.2fdeg)' % (
                '-' if angle is None else '%+.2fdeg' % angle,
                yellow_align_wz(angle, self.cfg),
                float(self.cfg['yellow_angle_align_deadband_deg'])),
        ])


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _build_page(client_fps):
    definitions = [dict({'key': key}, **value)
                   for key, value in STREAMS.items()]
    return PAGE_TEMPLATE.replace(
        '__STREAM_DEFINITIONS__',
        json.dumps(definitions, ensure_ascii=False, separators=(',', ':'))
    ).replace(
        '__CLIENT_PERIOD_MS__',
        str(int(round(1000.0 / max(float(client_fps), 0.1))))
    ).encode('utf-8')


def _make_handler(node, page):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'CyberDogStage2Vision/1.0'

        def log_message(self, format_string, *args):
            return

        def send_payload(self, status, content_type, payload, headers=None):
            try:
                self.connection.settimeout(2.0)
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store, no-cache')
                if headers:
                    for name, value in headers.items():
                        self.send_header(name, str(value))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)
            except OSError:
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == '/':
                self.send_payload(200, 'text/html; charset=utf-8', page)
                return
            if parsed.path == '/api/status':
                payload = json.dumps(
                    node.state.status(node), ensure_ascii=False,
                    separators=(',', ':')).encode('utf-8')
                self.send_payload(200, 'application/json; charset=utf-8',
                                  payload)
                return
            if parsed.path.startswith('/api/frame/'):
                key = parsed.path[len('/api/frame/'):]
                after = None
                values = parse_qs(parsed.query).get('after')
                if values:
                    try:
                        after = int(values[0])
                    except (TypeError, ValueError):
                        self.send_payload(400, 'text/plain', b'invalid after')
                        return
                snapshot = node.state.frame(key, after)
                if snapshot is None:
                    self.send_payload(204, 'image/jpeg', b'')
                    return
                sequence, image_bytes, age = snapshot
                if image_bytes is None:
                    self.send_payload(204, 'image/jpeg', b'',
                                      {'X-Frame-Id': sequence})
                    return
                self.send_payload(200, 'image/jpeg', image_bytes, {
                    'X-Frame-Id': sequence,
                    'X-Frame-Age': '%.3f' % age,
                })
                return
            self.send_payload(404, 'text/plain', b'not found')

    return Handler


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Read-only Stage-2 fisheye/RGB detection web preview')
    parser.add_argument('--platform', choices=('sim', 'real'), default='real')
    parser.add_argument('--dog-ns', default='auto')
    parser.add_argument('--rgb-topic')
    parser.add_argument('--depth-topic')
    parser.add_argument('--fisheye-left')
    parser.add_argument('--fisheye-right')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--max-fps', type=float, default=1.0)
    parser.add_argument('--client-fps', type=float, default=1.0)
    parser.add_argument('--stale-after', type=float, default=2.5)
    parser.add_argument('--processing-timeout', type=float, default=5.0)
    parser.add_argument('--opencv-threads', type=int, default=1)
    parser.add_argument('--jpeg-quality', type=int, default=75)
    args, ros_args = parser.parse_known_args(argv)
    if args.port < 1 or args.port > 65535:
        parser.error('--port must be between 1 and 65535')
    if args.max_fps <= 0 or args.client_fps <= 0:
        parser.error('--max-fps and --client-fps must be greater than 0')
    if args.stale_after <= 0:
        parser.error('--stale-after must be greater than 0')
    if args.processing_timeout <= 0:
        parser.error('--processing-timeout must be greater than 0')
    if args.opencv_threads < 1:
        parser.error('--opencv-threads must be at least 1')
    if args.jpeg_quality < 30 or args.jpeg_quality > 100:
        parser.error('--jpeg-quality must be between 30 and 100')
    return args, ros_args


def _resolve_topics(node, args):
    if str(args.dog_ns).lower() == 'auto':
        namespace = _discover_namespace(node, 3.0)
        if not namespace and node.platform == 'real':
            namespace = '/mi_desktop_48_b0_2d_7b_00_e2'
    else:
        namespace = _normalize_namespace(args.dog_ns)
    if node.platform == 'real':
        defaults = {
            'rgb': _join_topic(namespace, 'image_rgb'),
            'depth': _join_topic(namespace, 'camera/depth/image_rect_raw'),
            'fisheye_left': _join_topic(namespace, 'image_left'),
            'fisheye_right': _join_topic(namespace, 'image_right'),
        }
    else:
        defaults = {
            'rgb': '/rgb_camera/rgb_camera/image_raw',
            'depth': '/d435/depth/d435_depth/depth/image_raw',
            'fisheye_left': '/image_left',
            'fisheye_right': '/image_right',
        }
    overrides = {
        'rgb': args.rgb_topic,
        'depth': args.depth_topic,
        'fisheye_left': args.fisheye_left,
        'fisheye_right': args.fisheye_right,
    }
    parameters = {
        'rgb': 'rgb_topic',
        'depth': 'depth_topic',
        'fisheye_left': 'fisheye_left_topic',
        'fisheye_right': 'fisheye_right_topic',
    }
    topics = {}
    for name in PreviewState.CHANNELS:
        parameter = str(node.get_parameter(parameters[name]).value).strip()
        topics[name] = overrides[name] or parameter or defaults[name]
    return topics


def main(argv=None):
    args, ros_args = _parse_args(argv)
    rclpy.init(args=ros_args)
    node = Stage2VisionPreviewNode(args)
    server = None
    server_thread = None
    try:
        node.attach(_resolve_topics(node, args))
        page = _build_page(args.client_fps)
        server = _ThreadedHTTPServer(
            (args.bind, args.port), _make_handler(node, page))
        server_thread = threading.Thread(target=server.serve_forever,
                                         name='stage2-preview-http')
        server_thread.daemon = True
        server_thread.start()
        node.get_logger().info(
            'Stage2 read-only preview: http://%s:%d/ (max %.1f FPS)' % (
                args.bind, args.port, args.max_fps))
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print('[ERROR] %s' % error, file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=2.0)
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
