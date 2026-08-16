#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四赛段只读视觉调试网页。

订阅前向 RGB/Depth，在后台运行与 stage4_node 相同的检测器，并把综合标注、
各类 HSV/形态学掩膜和深度图发布到本机 HTTP 页面。此节点不创建机器人控制器，
也不发布任何运动命令，适合通过 SSH 端口转发在真机上调试。
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
# python3 control_node/stage4_vision_preview.py ...
if __package__ is None or __package__ == '':
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image

from control_node.stage4_node import (
    BallDetector,
    BarColorDetector,
    ColaDetector,
    FootballDetector,
    ObstacleBlueDetector,
    YellowDashedLineDetector,
    YellowHorizontalLineDetector,
)


STREAMS = collections.OrderedDict([
    ('bar_obstacle', {
        'label': '限高杆与障碍物', 'primary': True, 'withPanel': True}),
    ('bar_obstacle_debug', {
        'label': '限高杆/障碍物判定条件', 'primary': True, 'tall': True}),
    ('targets', {
        'label': 'RGB 目标检测（白球、可乐）',
        'primary': True, 'withPanel': True}),
    ('basketball_ai', {
        'label': 'AI 相机篮球检测',
        'primary': True, 'withPanel': True}),
    ('cola_debug', {'label': '可乐判定条件（通过与未通过）', 'primary': True}),
    ('yellow', {'label': '黄线检测', 'primary': True}),
])


PAGE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyberDog 第四赛段视觉调试</title>
<style>
:root{color-scheme:dark;--bg:#08111e;--panel:#111d2e;--line:#29384d;--text:#edf4fc;--muted:#91a5bd;--ok:#35d383;--warn:#ffc857;--bad:#ff6b7e;--accent:#63adff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;gap:14px;align-items:center;justify-content:space-between;padding:13px 18px;background:rgba(8,17,30,.95);border-bottom:1px solid var(--line)}
.title{font-size:20px;font-weight:760}.subtitle{color:var(--muted);font-size:12px}.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.pill,select,button{border:1px solid var(--line);border-radius:8px;background:#17253a;color:var(--text);padding:6px 10px}.pill{color:var(--muted)}button{cursor:pointer}
main{max-width:1800px;margin:auto;padding:14px}.notice{margin-bottom:12px;padding:10px 12px;border:1px solid #6a5221;border-radius:9px;background:#2b2415;color:#ffd878}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}.card.small{grid-column:span 1}.head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line)}.name{font-weight:700}.health{color:var(--warn)}.health.ok{color:var(--ok)}.health.bad{color:var(--bad)}.viewport{position:relative;aspect-ratio:4/3;background:#02060b;display:grid;place-items:center}.card.tall .viewport{aspect-ratio:1/1.42}.card.with-panel .viewport{aspect-ratio:1/1.04}.viewport img{display:block;width:100%;height:100%;object-fit:contain}.placeholder{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);background:rgba(2,6,11,.75)}.placeholder[hidden]{display:none}.meta{padding:8px 12px;color:var(--muted);font-size:12px}
.details{margin-top:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.detail{padding:10px 12px;background:var(--panel);border:1px solid var(--line);border-radius:10px}.detail b{display:block;margin-bottom:4px}.detail pre{margin:0;color:var(--muted);white-space:pre-wrap;word-break:break-word;font:12px/1.45 ui-monospace,monospace}
footer{max-width:1800px;margin:auto;padding:0 16px 18px;color:var(--muted);font-size:12px}@media(max-width:1200px){.grid{grid-template-columns:1fr}.details{grid-template-columns:1fr}.card.small{grid-column:auto}header{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<header><div><div class="title">CyberDog 第四赛段视觉调试</div><div class="subtitle">只订阅相机 · 不发送运动命令 · 页面适合 SSH 端口转发</div></div><div class="controls"><span id="summary" class="pill">正在连接…</span><label>刷新 <select id="rate"><option selected>1</option><option>2</option><option>3</option><option>5</option></select> FPS</label><button id="pause">暂停</button></div></header>
<main><div id="notice" class="notice" hidden></div><section id="grid" class="grid"></section><section class="details"><div class="detail"><b>相机</b><pre id="camera">等待数据</pre></div><div class="detail"><b>检测结果</b><pre id="detections">等待数据</pre></div><div class="detail"><b>参数模式</b><pre id="config">等待数据</pre></div></section></main>
<footer>画面分别显示：限高杆与障碍物、限高杆/障碍物逐条件判定、RGB 白球/可乐目标、AI 相机篮球与上沿触发参数、可乐逐条件判定、虚线与终点横向黄线。绿色竖线是图像中心，矩形表示 ROI；页面进入后台后会暂停拉图。</footer>
<script>
'use strict';
const defs=__STREAM_DEFINITIONS__,app={paused:false,hidden:document.hidden,period:__CLIENT_PERIOD_MS__,statusFails:0};
const grid=document.getElementById('grid'),summary=document.getElementById('summary'),notice=document.getElementById('notice'),rate=document.getElementById('rate'),pause=document.getElementById('pause');
const cards=defs.map(d=>{const e=document.createElement('article');e.className='card'+(d.primary?'':' small')+(d.tall?' tall':'')+(d.withPanel?' with-panel':'');e.innerHTML=`<div class="head"><span class="name">${d.label}</span><span class="health">等待首帧</span></div><div class="viewport"><img alt="${d.label}"><div class="placeholder">等待首帧…</div></div><div class="meta">帧号 <span class="seq">0</span> · 预览延迟 <span class="age">-</span></div>`;grid.appendChild(e);return {...d,e,img:e.querySelector('img'),health:e.querySelector('.health'),placeholder:e.querySelector('.placeholder'),seq:e.querySelector('.seq'),age:e.querySelector('.age'),last:null,url:null}});
const sleep=ms=>new Promise(r=>setTimeout(r,ms)),stopped=()=>app.paused||app.hidden;
async function get(url,ms){const c=new AbortController(),t=setTimeout(()=>c.abort(),ms);try{return await fetch(url,{cache:'no-store',signal:c.signal})}finally{clearTimeout(t)}}
function text(id,value){document.getElementById(id).textContent=value}
function fmtDet(d){if(!d)return 'none';const depth=d.depth_m==null?'?':d.depth_m.toFixed(2)+'m';return `${d.type} center=(${d.center[0]},${d.center[1]}) depth=${depth}${d.method?' method='+d.method:''}`}
async function statusLoop(){for(;;){if(stopped()){await sleep(200);continue}try{const r=await get('/api/status',1800);if(!r.ok)throw Error(r.status);const s=await r.json();app.statusFails=0;const age=s.rgb_age_s==null?'-':s.rgb_age_s.toFixed(2)+'s',aiAge=s.ai_age_s==null?'-':s.ai_age_s.toFixed(2)+'s';summary.textContent=`RGB ${s.rgb_received} 帧 · ${age} / AI ${s.ai_received} 帧 · ${aiAge}`;summary.style.color=s.health==='live'?'var(--ok)':s.health==='error'?'var(--bad)':'var(--warn)';notice.hidden=!s.warning;notice.textContent=s.warning||'';text('camera',`RGB: ${s.rgb_topic}\nDepth: ${s.depth_topic}\nAI: ${s.ai_topic}\n尺寸: ${s.rgb_shape||'-'} / ${s.depth_shape||'-'} / ${s.ai_shape||'-'}\n发布者: ${s.rgb_publishers} / ${s.depth_publishers} / ${s.ai_publishers}`);text('detections',(s.detections||[]).map(fmtDet).join('\n')||'当前没有通过过滤的目标');const duration=s.process_duration_s==null?'-':s.process_duration_s.toFixed(3)+'s';text('config',`platform=${s.platform}\nbar HSV=${s.bar_hsv}\n处理=${s.process_fps||0} FPS\n单帧耗时=${duration}\n当前步骤=${s.processing_step}\n丢弃旧帧=${s.dropped}`)}catch(e){app.statusFails++;if(app.statusFails>=3){summary.textContent='调试服务连接异常';summary.style.color='var(--bad)'}}await sleep(1000)}}
async function show(card,blob){const next=URL.createObjectURL(blob),old=card.url;await new Promise((ok,bad)=>{card.img.onload=ok;card.img.onerror=bad;card.img.src=next});card.url=next;if(old)URL.revokeObjectURL(old)}
async function frameLoop(card,delay){await sleep(delay);for(;;){if(stopped()){await sleep(180);continue}const started=performance.now();try{const q=card.last==null?'':`?after=${card.last}`;const r=await get(`/api/frame/${card.key}${q}`,3500);if(r.status===204){await sleep(80);continue}if(!r.ok)throw Error(r.status);await show(card,await r.blob());card.last=Number(r.headers.get('X-Frame-Id'));card.seq.textContent=card.last;card.age.textContent=Number(r.headers.get('X-Frame-Age')||0).toFixed(2)+'s';card.health.textContent='正常';card.health.className='health ok';card.placeholder.hidden=true}catch(e){card.health.textContent='拉图失败';card.health.className='health bad'}await sleep(Math.max(60,app.period-(performance.now()-started)))}}
rate.onchange=()=>{app.period=1000/Number(rate.value)};pause.onclick=()=>{app.paused=!app.paused;pause.textContent=app.paused?'恢复':'暂停'};document.addEventListener('visibilitychange',()=>{app.hidden=document.hidden});window.addEventListener('beforeunload',()=>cards.forEach(c=>{if(c.url)URL.revokeObjectURL(c.url)}));
statusLoop();cards.forEach((c,i)=>frameLoop(c,i*90));
</script>
</body></html>'''


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


def _bar_defaults(platform):
    if platform == 'real':
        hsv = {'h_min': 170, 'h_max': 10, 's_min': 80, 's_max': 255,
               'v_min': 50, 'v_max': 255, 'max_aspect_ratio': 6.0}
    else:
        hsv = {'h_min': 85, 'h_max': 100, 's_min': 15, 's_max': 45,
               'v_min': 35, 'v_max': 80, 'max_aspect_ratio': 50.0}
    hsv.update({
        'roi_x_ratio_min': 0.0, 'roi_x_ratio_max': 1.0,
        'roi_y_ratio_min': 0.0, 'roi_y_ratio_max': 1.0,
        'open_kernel': 3, 'close_kernel_h': 7, 'close_kernel_w': 11,
        'min_area': 1000, 'min_width': 15, 'max_height': 1000,
        'min_aspect_ratio': 1.5,
        'max_center_y_ratio_in_roi': 1.0,
        'center_weight_base': 0.3, 'center_weight_gain': 0.7,
        'structure_check_enabled': platform == 'real',
        'structure_inner_ratio': 0.40,
        'structure_max_inner_red_ratio': 0.35,
        'structure_min_depth_gap_m': 0.20,
        'structure_near_bypass_distance_m': 1.00,
        'structure_near_bypass_width_ratio': 0.40,
        'structure_min_depth_pixels': 10,
    })
    return hsv


OBSTACLE_DEFAULTS = {
    'use_depth_filter': True,
    'roi_x_ratio_min': 0.0, 'roi_x_ratio_max': 1.0,
    'roi_y_ratio_min': 0.0, 'roi_y_ratio_max': 1.0,
    'h_min': 90, 'h_max': 140, 's_min': 60, 's_max': 255,
    'v_min': 40, 'v_max': 255, 'depth_min_m': 0.05, 'depth_max_m': 1.50,
    'open_kernel': 3, 'close_kernel': 5,
    'min_area': 150, 'max_area': 1000000,
    'min_width': 10, 'min_height': 10,
    'min_aspect_ratio': 0.0, 'max_aspect_ratio': 4.5,
    'min_bottom_y_ratio_in_roi': 0.2, 'min_valid_depth_ratio': 0.20,
    'min_near_depth_ratio': 0.35, 'bbox_depth_percentile': 50.0,
    'bbox_depth_margin_m': 0.10, 'bbox_depth_min_pixels': 40,
}

RGB_TRIGGER_DEFAULTS = collections.OrderedDict([
    ('bar_search_slow_top_y_ratio', 0.18),
    ('bar_trigger_top_y_ratio', 0.08),
    ('obstacle_approach_pitch_bottom_y_ratio', 0.75),
    ('obstacle_trigger_bottom_y_ratio', 0.90),
    ('post_hit_obstacle_trigger_bottom_y_ratio', 0.95),
    ('cola_slow_cap_area_ratio', 0.0015),
    ('cola_hit_cap_area_ratio', 0.0040),
    ('white_ball_slow_area_ratio', 0.015),
    ('white_ball_hit_area_ratio', 0.040),
    ('white_ball_slow_radius_px', 35.0),
    ('white_ball_hit_radius_px', 60.0),
])

BASKETBALL_AI_DEFAULTS = collections.OrderedDict([
    ('basketball_ai_roi_x_min_ratio', 0.20),
    ('basketball_ai_roi_x_max_ratio', 0.80),
    ('basketball_ai_roi_y_min_ratio', 0.05),
    ('basketball_ai_roi_y_max_ratio', 0.90),
    ('basketball_ai_max_age_s', 0.50),
    ('basketball_top_slow_y_ratio', 0.35),
    ('basketball_top_trigger_y_ratio', 0.25),
    ('basketball_top_trigger_confirm_frames', 3),
])

YELLOW_DEFAULTS = {
    'h_min': 18, 'h_max': 45, 's_min': 70, 's_max': 255,
    'v_min': 70, 'v_max': 255, 'roi_x_ratio_min': 0.0,
    'roi_x_ratio_max': 1.0, 'roi_y_ratio_min': 0.60,
    'roi_y_ratio_max': 1.0, 'open_kernel': 3, 'min_area': 50,
    'max_area': 4000, 'min_width': 3, 'min_height': 5,
    'dash_close_kernel_h': 3, 'dash_close_kernel_w': 5,
    'dash_min_segments': 2, 'dash_min_total_span_y': 20,
    'dash_max_adjacent_x_diff': 110, 'dash_max_gap_y': 3000,
    'dash_min_gap_y': -10, 'dash_max_total_x_range': 5000,
    'dash_segment_max_aspect_ratio': 10.0,
    'dash_segment_max_long_side': 200, 'dash_duplicate_iou_thresh': 0.35,
    'dash_duplicate_center_x_thresh': 30, 'max_dashed_lines': 2,
}

FINAL_YELLOW_DEFAULTS = {
    'h_min': 18, 'h_max': 45, 's_min': 70, 's_max': 255,
    'v_min': 70, 'v_max': 255, 'roi_x_ratio_min': 0.30,
    'roi_x_ratio_max': 0.70, 'roi_y_ratio_min': 0.50,
    'roi_y_ratio_max': 1.0, 'open_kernel': 3,
    'close_kernel_h': 5, 'close_kernel_w': 11, 'min_area': 3000,
    'min_width': 20, 'min_height': 3, 'min_width_ratio': 0.70,
    'min_wh_ratio': 1.5, 'max_tilt_deg': 35.0,
    'center_tolerance_ratio': 0.60,
}

BLUE_BALL_DEFAULTS = {
    'h_min': 90, 'h_max': 135, 's_min': 80, 's_max': 255,
    'v_min': 40, 'v_max': 255, 'roi_x_ratio_min': 0.0,
    'roi_x_ratio_max': 1.0, 'roi_y_ratio_min': 0.0,
    'roi_y_ratio_max': 1.0, 'open_kernel': 3, 'close_kernel': 5,
    'min_area': 80, 'max_area': 5000000, 'min_radius': 5.0,
    'max_radius': 200.0, 'min_circularity': 0.82,
    'min_wh_ratio': 0.75, 'max_wh_ratio': 1.33,
    'max_center_y_ratio_in_roi': 1.0,
    'center_weight_base': 0.3, 'center_weight_gain': 0.7,
    'radius_score_gain': 10.0,
}

WHITE_BALL_DEFAULTS = dict(BLUE_BALL_DEFAULTS)
WHITE_BALL_DEFAULTS.update({
    'h_min': 0, 'h_max': 20, 's_min': 0, 's_max': 20,
    'v_min': 95, 'v_max': 255, 'max_area': 50000,
    'min_radius': 10.0, 'max_radius': 150.0,
    'min_circularity': 0.55, 'min_wh_ratio': 0.60,
    'max_wh_ratio': 1.40,
})

COLA_DEFAULTS = {
    'roi_x_ratio_min': 0.0, 'roi_x_ratio_max': 1.0,
    'roi_y_ratio_min': 0.0, 'roi_y_ratio_max': 1.0,
    'cap_only_mode': False,
    'enable_dark_shape': False,
    'dark_v_max': 110, 'dark_s_min': 35, 'very_dark_v_max': 55,
    'open_kernel': 3, 'close_kernel': 7, 'min_area': 120,
    'max_area': 100000, 'min_width': 6, 'max_width': 300,
    'min_height': 24, 'max_height': 470, 'min_aspect': 1.90,
    'max_aspect': 5.5, 'target_aspect': 3.0, 'min_fill_ratio': 0.28,
    'max_fill_ratio': 0.95, 'min_symmetry': 0.50,
    'min_shoulder_ratio': 0.18, 'max_shoulder_ratio': 0.95,
    'min_solidity': 0.68, 'min_bottom_ratio': 0.28,
    'full_size_area': 5000.0, 'min_score': 55.0,
    'cap_h_low_max': 15, 'cap_h_high_min': 165,
    'cap_s_min': 75, 'cap_v_min': 55,
    'cap_min_area_ratio': 0.0003, 'cap_max_area_ratio': 0.01,
    'cap_min_aspect': 0.45, 'cap_max_aspect': 2.20,
    'cap_body_min_area': 60.0, 'cap_body_area_gain': 3.5,
    'cap_body_min_width': 5, 'cap_body_min_height': 18,
    'cap_bottle_min_aspect': 1.60, 'cap_bottle_max_aspect': 5.5,
    'center_smoothing_alpha': 0.35, 'max_center_jump_ratio': 0.25,
}


def _declare_config(node, prefix, defaults):
    for key, value in defaults.items():
        node.declare_parameter('%s.%s' % (prefix, key), value)


def _read_config(node, prefix, defaults):
    return {
        key: node.get_parameter('%s.%s' % (prefix, key)).value
        for key in defaults
    }


def _depth_meters(depth):
    if depth is None:
        return None
    if depth.dtype == np.uint16:
        result = depth.astype(np.float32) / 1000.0
    else:
        result = depth.astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def _depth_at(depth_m, center, rgb_shape):
    if depth_m is None:
        return None
    dh, dw = depth_m.shape[:2]
    rh, rw = rgb_shape[:2]
    cx, cy = center
    dx = int(cx * dw / max(rw, 1))
    dy = int(cy * dh / max(rh, 1))
    patch = depth_m[max(0, dy - 3):min(dh, dy + 4),
                    max(0, dx - 3):min(dw, dx + 4)]
    valid = patch[np.isfinite(patch)]
    valid = valid[(valid > 0.05) & (valid < 10.0)]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, 20))


def _encode_jpeg(image, quality):
    ok, encoded = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError('OpenCV JPEG encoding failed')
    return encoded.tobytes()


def _put_label(image, text, point, color, scale=0.48):
    x, y = int(point[0]), int(point[1])
    cv2.putText(image, str(text), (x, max(18, y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, str(text), (x, max(18, y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _condition(name, value, rule, passed):
    return {
        'name': str(name),
        'value': str(value),
        'rule': str(rule),
        'passed': bool(passed),
    }


def _bar_debug_records(frame, depth, detector, cfg):
    """Mirror BarColorDetector checks and retain rejected contours."""
    (roi_box, roi) = detector._roi(frame)
    x1, y1, _, _ = roi_box
    roi_h, roi_w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detector.lower_bar, detector.upper_bar)
    if detector.hue_wraps:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, detector.lower_bar_high,
                        detector.upper_bar_high),
        )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, detector.kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, detector.kernel_close)

    depth_m = detector._depth_to_meters(depth)
    roi_depth = None
    if depth_m is not None:
        if depth_m.shape[:2] != frame.shape[:2]:
            depth_m = cv2.resize(
                depth_m, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        roi_depth = depth_m[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]

    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    records = []
    min_display_area = max(20.0, float(cfg['min_area']) * 0.15)
    roi_center_x = roi_w * 0.5
    for contour in contours:
        area = float(cv2.contourArea(contour))
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw <= 0 or rh <= 0 or area < min_display_area:
            continue

        aspect = rw / float(rh)
        center_y_ratio = (ry + rh * 0.5) / float(max(roi_h, 1))
        width_ratio = rw / float(max(frame.shape[1], 1))
        center_bonus = 1.0 - abs(
            (rx + rw * 0.5) - roi_center_x) / max(roi_center_x, 1.0)
        checks = [
            _condition('area', '%.0f' % area,
                       '>=%.0f' % cfg['min_area'], area >= cfg['min_area']),
            _condition('width', rw, '>=%d' % cfg['min_width'],
                       rw >= cfg['min_width']),
            _condition('height', rh, '<=%d' % cfg['max_height'],
                       rh <= cfg['max_height']),
            _condition('aspect', '%.2f' % aspect, '%.2f..%.2f' % (
                cfg['min_aspect_ratio'], cfg['max_aspect_ratio']),
                cfg['min_aspect_ratio'] <= aspect <= cfg['max_aspect_ratio']),
            _condition('center_y', '%.3f' % center_y_ratio, '<=%.3f' % (
                cfg['max_center_y_ratio_in_roi']),
                center_y_ratio <= cfg['max_center_y_ratio_in_roi']),
        ]

        structure_ok = True
        if cfg.get('structure_check_enabled', False):
            depth_patch = (roi_depth[ry:ry + rh, rx:rx + rw]
                           if roi_depth is not None else None)
            structure_ok, info = detector._check_structure(
                mask[ry:ry + rh, rx:rx + rw], depth_patch,
                frame.shape[1])
            bar_depth = info.get('bar_surface_depth_m')
            depth_gap = info.get('structure_depth_gap_m')
            inner_red = float(info.get('inner_red_ratio', 1.0))
            near_depth = (
                bar_depth is not None
                and bar_depth <= cfg['structure_near_bypass_distance_m'])
            near_width = width_ratio >= cfg['structure_near_bypass_width_ratio']
            hole_ok = inner_red <= cfg['structure_max_inner_red_ratio']
            gap_ok = (
                depth_gap is not None
                and depth_gap >= cfg['structure_min_depth_gap_m'])
            checks.extend([
                _condition('alt_near_d', '-' if bar_depth is None else
                           '%.2fm' % bar_depth, '<=%.2fm' %
                           cfg['structure_near_bypass_distance_m'], near_depth),
                _condition('alt_near_w', '%.3f' % width_ratio, '>=%.3f' %
                           cfg['structure_near_bypass_width_ratio'], near_width),
                _condition('alt_hole', '%.3f' % inner_red, '<=%.3f' %
                           cfg['structure_max_inner_red_ratio'], hole_ok),
                _condition('alt_depth_gap', '-' if depth_gap is None else
                           '%.2fm' % depth_gap, '>=%.2fm' %
                           cfg['structure_min_depth_gap_m'], gap_ok),
                _condition('structure_any',
                           'PASS' if structure_ok else 'FAIL',
                           'any alt PASS', structure_ok),
            ])

        accepted = all(item['passed'] for item in checks[:5]) and structure_ok
        records.append({
            'bbox': (x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh),
            'area': area,
            'score': float(max(center_bonus, 0.0)),
            'checks': checks,
            'accepted': accepted,
        })
    return records


def _obstacle_debug_records(debug_infos, cfg):
    records = []
    for info in debug_infos:
        _, _, width, height = info['bbox_roi']
        area = float(info['area'])
        aspect = float(info['aspect_ratio'])
        bottom = float(info['bottom_y_ratio'])
        checks = [
            _condition('area', '%.0f' % area,
                       '%.0f..%.0f' % (cfg['min_area'], cfg['max_area']),
                       cfg['min_area'] <= area <= cfg['max_area']),
            _condition('width', width, '>=%d' % cfg['min_width'],
                       width >= cfg['min_width']),
            _condition('height', height, '>=%d' % cfg['min_height'],
                       height >= cfg['min_height']),
            _condition('aspect', '%.2f' % aspect, '%.2f..%.2f' % (
                cfg['min_aspect_ratio'], cfg['max_aspect_ratio']),
                cfg['min_aspect_ratio'] <= aspect <= cfg['max_aspect_ratio']),
            _condition('bottom_y', '%.3f' % bottom, '>=%.3f' %
                       cfg['min_bottom_y_ratio_in_roi'],
                       bottom >= cfg['min_bottom_y_ratio_in_roi']),
        ]
        if cfg.get('use_depth_filter', True):
            valid_ratio = info.get('valid_depth_ratio')
            near_ratio = info.get('near_depth_ratio')
            checks.extend([
                _condition('valid_depth', '-' if valid_ratio is None else
                           '%.3f' % valid_ratio, '>=%.3f' %
                           cfg['min_valid_depth_ratio'],
                           valid_ratio is not None and
                           valid_ratio >= cfg['min_valid_depth_ratio']),
                _condition('near_depth', '-' if near_ratio is None else
                           '%.3f' % near_ratio, '>=%.3f' %
                           cfg['min_near_depth_ratio'],
                           near_ratio is not None and
                           near_ratio >= cfg['min_near_depth_ratio']),
            ])
        records.append({
            'bbox': tuple(info['bbox_img']),
            'area': area,
            'score': area + 100.0 * bottom,
            'checks': checks,
            'accepted': bool(info['passed']),
        })
    return records


def _draw_condition_panel(image, record, title, x, y, width):
    checks = record['checks'] if record is not None else []
    height = 52 + 15 * len(checks)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y),
                  (min(image.shape[1] - 5, x + width),
                   min(image.shape[0] - 5, y + height)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    _put_label(image, title, (x + 7, y + 18), (255, 255, 255), 0.40)
    if record is None:
        _put_label(image, 'NO COLOR CONTOUR TO EVALUATE', (x + 7, y + 40),
                   (0, 165, 255), 0.38)
        return
    result_color = (60, 230, 60) if record['accepted'] else (60, 60, 255)
    _put_label(image, '%s  score=%.1f' % (
        'PASS' if record['accepted'] else 'FAIL', record['score']),
        (x + 7, y + 39), result_color, 0.42)
    line_y = y + 58
    for item in checks:
        color = (60, 230, 60) if item['passed'] else (60, 60, 255)
        _put_label(image, '%s %-13s %s rule:%s' % (
            '+' if item['passed'] else '-', item['name'],
            item['value'], item['rule']), (x + 7, line_y), color, 0.31)
        line_y += 15


def _bar_obstacle_debug_view(frame, depth, bar_detector, bar_cfg,
                             selected_bar, obstacle_infos, obstacle_cfg):
    view = frame.copy()
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)
    bar_records = _bar_debug_records(
        frame, depth, bar_detector, bar_cfg)
    obstacle_records = _obstacle_debug_records(obstacle_infos, obstacle_cfg)

    def prepare(records, selected_bbox=None, obstacle_order=False):
        if obstacle_order:
            # 与控制节点一致：通过候选优先，再按外接框底边从低到高、面积从大到小排序。
            records.sort(key=lambda item: (
                item['accepted'], int(item['bbox'][3]), item['area']),
                reverse=True)
        else:
            records.sort(key=lambda item: (
                item['accepted'],
                sum(check['passed'] for check in item['checks']),
                item['score'], item['area']), reverse=True)
        if selected_bbox is not None:
            for index, record in enumerate(records):
                if tuple(record['bbox']) == tuple(selected_bbox):
                    records.insert(0, records.pop(index))
                    break
        # Keep the two best visible candidates even when they failed.  The
        # panel below will then show every measured value, threshold and failed
        # condition instead of becoming empty when the detector returns None.
        return records[:2]

    bar_bbox = selected_bar.bbox_img if selected_bar is not None else None
    bar_details = prepare(bar_records, bar_bbox)
    obstacle_details = prepare(obstacle_records, obstacle_order=True)

    for prefix, records in (('B', bar_records), ('O', obstacle_records)):
        for record in records[:10]:
            x1, y1, x2, y2 = [int(value) for value in record['bbox']]
            color = (0, 220, 0) if record['accepted'] else (0, 0, 255)
            details = bar_details if prefix == 'B' else obstacle_details
            thickness = 3 if any(record is item for item in details) else 1
            cv2.rectangle(view, (x1, y1), (x2, y2), color, thickness)

    # Only accepted obstacle records may form the control pair.  Failed
    # candidates stay visible in the panels but must not look like a selected
    # two-obstacle target in the image overlay.
    selected_obstacles = [
        record for record in obstacle_details if record['accepted']]
    if len(selected_obstacles) == 2:
        centers = []
        for record in selected_obstacles:
            x1, y1, x2, y2 = record['bbox']
            centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        pair_center = (
            int((centers[0][0] + centers[1][0]) / 2),
            int((centers[0][1] + centers[1][1]) / 2),
        )
        cv2.line(view, centers[0], centers[1], (0, 255, 255), 2)
        cv2.circle(view, pair_center, 7, (0, 255, 255), -1)

    all_details = bar_details + obstacle_details
    max_checks = max(
        [len(detail['checks']) for detail in all_details] or [0])
    panel_row_height = max(115, 62 + 15 * max_checks)
    panel_height = panel_row_height * 2 + 15
    canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    canvas[:] = (8, 17, 30)
    canvas[:h, :w] = view
    cv2.line(canvas, (0, h), (w - 1, h), (65, 82, 105), 1)
    panel_width = max(250, w // 2 - 8)
    _draw_condition_panel(
        canvas, bar_details[0] if len(bar_details) > 0 else None,
        'BAR CANDIDATE 1 (PASS / FAIL)', 5, h + 5, panel_width)
    _draw_condition_panel(
        canvas, bar_details[1] if len(bar_details) > 1 else None,
        'BAR CANDIDATE 2 (PASS / FAIL)', w // 2 + 3, h + 5, panel_width)
    obstacle_row_y = h + 10 + panel_row_height
    _draw_condition_panel(
        canvas, obstacle_details[0] if len(obstacle_details) > 0 else None,
        'OBSTACLE CANDIDATE 1 (PASS / FAIL)', 5, obstacle_row_y, panel_width)
    _draw_condition_panel(
        canvas, obstacle_details[1] if len(obstacle_details) > 1 else None,
        'OBSTACLE CANDIDATE 2 (PASS / FAIL)', w // 2 + 3, obstacle_row_y, panel_width)
    return canvas


def _append_status_panel(image, title, lines, panel_height):
    h, w = image.shape[:2]
    canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
    canvas[:] = (8, 17, 30)
    canvas[:h, :w] = image
    cv2.line(canvas, (0, h), (w - 1, h), (65, 82, 105), 1)
    _put_label(canvas, title, (9, h + 22), (255, 255, 255), 0.45)
    line_y = h + 44
    for text, color in lines:
        _put_label(canvas, text, (9, line_y), color, 0.36)
        line_y += 18
    return canvas


def _bar_obstacle_trigger_view(image, bar, obstacles, cfg):
    """Draw bar top-edge and obstacle bottom-edge trigger lines."""
    view = image.copy()
    h, w = view.shape[:2]
    rules = [
        ('BAR slow', cfg['bar_search_slow_top_y_ratio'],
         (0, 255, 255), 'top', '<='),
        ('BAR trigger', cfg['bar_trigger_top_y_ratio'],
         (0, 0, 255), 'top', '<='),
        ('OBS pitch', cfg['obstacle_approach_pitch_bottom_y_ratio'],
         (255, 255, 0), 'bottom', '>='),
        ('OBS dash', cfg['obstacle_trigger_bottom_y_ratio'],
         (0, 165, 255), 'bottom', '>='),
        ('OBS post-hit', cfg['post_hit_obstacle_trigger_bottom_y_ratio'],
         (255, 0, 255), 'bottom', '>='),
    ]
    for _, ratio, color, _, _ in rules:
        y = max(0, min(h - 1, int(round(h * ratio))))
        cv2.line(view, (0, y), (w - 1, y), color, 1)

    bar_top = (float(bar.bbox_img[1]) / float(max(h, 1))
               if bar is not None else None)
    selected_obstacles = list(obstacles[:2])
    obstacle_bottoms = [
        float(det.bbox_img[3]) / float(max(h, 1))
        for det in selected_obstacles
    ]
    obstacle_bottom = max(obstacle_bottoms) if obstacle_bottoms else None

    def yes_no(value, threshold, comparison):
        if value is None:
            return 'NO'
        reached = value <= threshold if comparison == '<=' else value >= threshold
        return 'YES' if reached else 'NO'

    def top_text(value):
        if value is None:
            return '-'
        return '%dpx/%.3f' % (int(round(value * h)), value)

    lines = []
    for name, threshold, color, edge, comparison in rules:
        current = bar_top if edge == 'top' else obstacle_bottom
        current_text = top_text(current)
        lines.append((
            '%-12s %s=%s  rule:%s%s%.3f  reached=%s' % (
                name, edge, current_text, edge, comparison, threshold,
                yes_no(current, threshold, comparison)),
            color,
        ))
    if obstacle_bottoms:
        lines.append((
            'OBS selected bottoms=' + ', '.join(top_text(value)
                                                 for value in obstacle_bottoms)
            + '  control=max(bottom)',
            (210, 210, 210),
        ))
    else:
        lines.append(('OBS selected bottoms=none', (210, 210, 210)))
    return _append_status_panel(
        view, 'RGB EDGE TRIGGERS (BAR: smaller top; OBS: larger bottom = closer)',
        lines, 160)


def _target_trigger_view(image, targets, cfg):
    """Show current area/radius against the RGB slow/hit thresholds."""
    by_type = {det.det_type: det for det in targets}
    lines = []

    cola = by_type.get('cola')
    cola_area = (float(cola.extra.get('cap_area_ratio', 0.0))
                 if cola is not None else None)
    cola_area_px = (float(cola.extra.get('area', 0.0))
                    if cola is not None else None)
    cola_slow = cfg['cola_slow_cap_area_ratio']
    cola_hit = cfg['cola_hit_cap_area_ratio']
    lines.append((
        'COLA cap_area=%s  ratio=%s  slow>=%.4f:%s  hit>=%.4f:%s' % (
            '-' if cola_area_px is None else '%.0fpx2' % cola_area_px,
            '-' if cola_area is None else '%.5f' % cola_area,
            cola_slow, 'YES' if cola_area is not None and cola_area >= cola_slow else 'NO',
            cola_hit, 'YES' if cola_area is not None and cola_area >= cola_hit else 'NO'),
        (0, 0, 255),
    ))

    for det_type, label, color in (
            ('white_ball', 'WHITE', (255, 255, 255)),):
        det = by_type.get(det_type)
        area = (float(det.extra.get('area_ratio', 0.0))
                if det is not None else None)
        area_px = (float(det.extra.get('area', 0.0))
                   if det is not None else None)
        radius = (float(det.extra.get('radius', 0.0))
                  if det is not None else None)
        slow_area = cfg[det_type + '_slow_area_ratio']
        hit_area = cfg[det_type + '_hit_area_ratio']
        slow_radius = cfg[det_type + '_slow_radius_px']
        hit_radius = cfg[det_type + '_hit_radius_px']
        slow_ok = (area is not None and radius is not None
                   and area >= slow_area and radius >= slow_radius)
        hit_ok = (area is not None and radius is not None
                  and area >= hit_area and radius >= hit_radius)
        lines.append((
            '%s area=%s ratio=%s radius=%s' % (
                label,
                '-' if area_px is None else '%.0fpx2' % area_px,
                '-' if area is None else '%.4f' % area,
                '-' if radius is None else '%.1fpx' % radius),
            color,
        ))
        lines.append((
            '  slow: area>=%.3f & r>=%.1f => %s; '
            'hit: area>=%.3f & r>=%.1f => %s' % (
                slow_area, slow_radius, 'YES' if slow_ok else 'NO',
                hit_area, hit_radius, 'YES' if hit_ok else 'NO'),
            color,
        ))
    return _append_status_panel(
        image, 'RGB TARGET SLOW / HIT TRIGGERS', lines, 145)


def _basketball_ai_view(image, det, detector_cfg, trigger_cfg, topic,
                        frame_age_sec, trigger_count):
    """Draw AI-camera basketball ROI, top-edge thresholds and parameters."""
    view = image.copy()
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h - 1), (0, 255, 0), 1)
    _roi_box(view, detector_cfg, (0, 255, 0), 'BASKETBALL AI')
    slow_ratio = trigger_cfg['basketball_top_slow_y_ratio']
    upright_ratio = trigger_cfg['basketball_top_trigger_y_ratio']
    slow_y = max(0, min(h - 1, int(round(h * slow_ratio))))
    upright_y = max(0, min(h - 1, int(round(h * upright_ratio))))
    cv2.line(view, (0, slow_y), (w - 1, slow_y), (0, 255, 255), 1)
    cv2.line(view, (0, upright_y), (w - 1, upright_y), (0, 0, 255), 1)
    _put_label(view, 'SLOW top<=%.3f' % slow_ratio,
               (5, min(h - 5, slow_y + 16)), (0, 255, 255), 0.40)
    _put_label(view, 'UPRIGHT top<=%.3f' % upright_ratio,
               (5, max(16, upright_y - 5)), (0, 0, 255), 0.40)

    top_ratio = None
    if det is not None:
        x1, y1, x2, _ = [int(value) for value in det.bbox_img]
        top_ratio = float(y1) / float(max(h, 1))
        _draw_detection(
            view, det, (255, 100, 0),
            'BLUE BALL top=%dpx/%.3f' % (y1, top_ratio), None)
        cv2.line(view, (x1, y1), (x2, y1), (255, 0, 255), 2)

    max_age = trigger_cfg['basketball_ai_max_age_s']
    fresh = frame_age_sec is not None and frame_age_sec <= max_age
    freshness = ('FRESH' if fresh else
                 ('NO FRAME' if frame_age_sec is None else 'STALE'))
    confirm_frames = int(trigger_cfg['basketball_top_trigger_confirm_frames'])
    slow_ok = top_ratio is not None and top_ratio <= slow_ratio
    upright_ok = top_ratio is not None and top_ratio <= upright_ratio
    lines = [
        ('topic=%s' % topic, (210, 210, 210)),
        ('frame age=%s  max=%.2fs  state=%s' % (
            '-' if frame_age_sec is None else '%.3fs' % frame_age_sec,
            max_age, freshness),
         (0, 255, 0) if fresh else (0, 0, 255)),
        ('ROI x=%.2f..%.2f  y=%.2f..%.2f' % (
            detector_cfg['roi_x_ratio_min'], detector_cfg['roi_x_ratio_max'],
            detector_cfg['roi_y_ratio_min'], detector_cfg['roi_y_ratio_max']),
         (0, 255, 0)),
        ('HSV H=%d..%d S=%d..%d V=%d..%d  min area=%.0f r=%.1f circ=%.2f' % (
            detector_cfg['h_min'], detector_cfg['h_max'],
            detector_cfg['s_min'], detector_cfg['s_max'],
            detector_cfg['v_min'], detector_cfg['v_max'],
            detector_cfg['min_area'], detector_cfg['min_radius'],
            detector_cfg['min_circularity']),
         (255, 100, 0)),
        ('top=%s  slow<=%.3f:%s  upright<=%.3f:%s' % (
            '-' if top_ratio is None else '%.3f' % top_ratio,
            slow_ratio, 'YES' if slow_ok else 'NO',
            upright_ratio, 'YES' if upright_ok else 'NO'),
         (255, 0, 255)),
        ('upright confirm=%d/%d  trigger=%s' % (
            trigger_count, confirm_frames,
            'YES' if trigger_count >= confirm_frames else 'NO'),
         (0, 0, 255) if trigger_count >= confirm_frames else (0, 255, 255)),
    ]
    if det is None:
        lines.append(('basketball=not detected', (180, 180, 180)))
    else:
        lines.append((
            'area=%.0fpx2 ratio=%.5f radius=%.1fpx circularity=%.3f' % (
                float(det.extra.get('area', 0.0)),
                float(det.extra.get('area_ratio', 0.0)),
                float(det.extra.get('radius', 0.0)),
                float(det.extra.get('circularity', 0.0))),
            (255, 100, 0),
        ))
    return _append_status_panel(
        view, 'AI BASKETBALL (top edge: smaller ratio = higher in image)',
        lines, 185)


def _draw_detection(image, det, color, label, depth_m):
    x1, y1, x2, y2 = [int(value) for value in det.bbox_img]
    cx, cy = [int(value) for value in det.center_img]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    cv2.circle(image, (cx, cy), 3, color, -1)
    if depth_m is None:
        text = label
    else:
        text = '%s d=%.2fm' % (label, depth_m)
    _put_label(image, text, (x1, y1 - 7), color)


def _roi_box(image, cfg, color, label):
    h, w = image.shape[:2]
    x1 = int(w * cfg['roi_x_ratio_min'])
    x2 = int(w * cfg['roi_x_ratio_max'])
    y1 = int(h * cfg['roi_y_ratio_min'])
    y2 = int(h * cfg['roi_y_ratio_max'])
    cv2.rectangle(image, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)),
                  color, 1, cv2.LINE_AA)
    _put_label(image, label + ' ROI', (x1 + 3, y1 + 17), color, 0.42)


def _full_hsv_mask(frame, cfg, open_shape=None, close_shape=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], np.uint8)
    upper = np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], np.uint8)
    if cfg['h_min'] <= cfg['h_max']:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        first = cv2.inRange(
            hsv, np.array([cfg['h_min'], cfg['s_min'], cfg['v_min']], np.uint8),
            np.array([179, cfg['s_max'], cfg['v_max']], np.uint8))
        second = cv2.inRange(
            hsv, np.array([0, cfg['s_min'], cfg['v_min']], np.uint8),
            np.array([cfg['h_max'], cfg['s_max'], cfg['v_max']], np.uint8))
        mask = cv2.bitwise_or(first, second)
    if open_shape:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones(open_shape, np.uint8))
    if close_shape:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones(close_shape, np.uint8))
    return mask


def _roi_mask(frame, cfg, mask_builder):
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(w * cfg['roi_x_ratio_min'])))
    x2 = max(x1 + 1, min(w, int(w * cfg['roi_x_ratio_max'])))
    y1 = max(0, min(h - 1, int(h * cfg['roi_y_ratio_min'])))
    y2 = max(y1 + 1, min(h, int(h * cfg['roi_y_ratio_max'])))
    result = np.zeros((h, w), np.uint8)
    result[y1:y2, x1:x2] = mask_builder(frame[y1:y2, x1:x2])
    return result


def _cola_masks(frame, cfg):
    def build(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        dark = np.where(
            ((value <= cfg['dark_v_max']) & (saturation >= cfg['dark_s_min'])) |
            (value <= cfg['very_dark_v_max']), 255, 0).astype(np.uint8)
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN,
            np.ones((cfg['open_kernel'], cfg['open_kernel']), np.uint8))
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_CLOSE,
            np.ones((cfg['close_kernel'], cfg['close_kernel']), np.uint8))
        low = cv2.inRange(
            hsv, np.array([0, cfg['cap_s_min'], cfg['cap_v_min']], np.uint8),
            np.array([cfg['cap_h_low_max'], 255, 255], np.uint8))
        high = cv2.inRange(
            hsv, np.array([cfg['cap_h_high_min'], cfg['cap_s_min'],
                           cfg['cap_v_min']], np.uint8),
            np.array([179, 255, 255], np.uint8))
        cap = cv2.morphologyEx(cv2.bitwise_or(low, high), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8))
        return dark, cap

    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(w * cfg['roi_x_ratio_min'])))
    x2 = max(x1 + 1, min(w, int(w * cfg['roi_x_ratio_max'])))
    y1 = max(0, min(h - 1, int(h * cfg['roi_y_ratio_min'])))
    y2 = max(y1 + 1, min(h, int(h * cfg['roi_y_ratio_max'])))
    dark_full = np.zeros((h, w), np.uint8)
    cap_full = np.zeros((h, w), np.uint8)
    dark, cap = build(frame[y1:y2, x1:x2])
    dark_full[y1:y2, x1:x2] = dark
    cap_full[y1:y2, x1:x2] = cap
    return dark_full, cap_full


def _cola_debug_candidates(frame, cfg):
    """Return visible cola candidates together with every pass/fail check."""
    frame_h, frame_w = frame.shape[:2]
    x1 = max(0, min(frame_w - 1,
                    int(frame_w * cfg['roi_x_ratio_min'])))
    x2 = max(x1 + 1, min(frame_w,
                         int(frame_w * cfg['roi_x_ratio_max'])))
    y1 = max(0, min(frame_h - 1,
                    int(frame_h * cfg['roi_y_ratio_min'])))
    y2 = max(y1 + 1, min(frame_h,
                         int(frame_h * cfg['roi_y_ratio_max'])))
    roi_h, roi_w = y2 - y1, x2 - x1
    roi_area = float(max(roi_h * roi_w, 1))
    dark_full, cap_full = _cola_masks(frame, cfg)
    dark_mask = dark_full[y1:y2, x1:x2]
    cap_mask = cap_full[y1:y2, x1:x2]
    records = []

    def contours(mask):
        return cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    def check(items, name, value, rule, passed):
        items.append({
            'name': str(name),
            'value': str(value),
            'rule': str(rule),
            'passed': bool(passed),
        })

    def row_widths(binary):
        widths = []
        for row in binary:
            xs = np.flatnonzero(row)
            widths.append(0.0 if xs.size == 0 else
                          float(xs[-1] - xs[0] + 1))
        return np.asarray(widths, dtype=np.float32)

    def median_positive(values):
        values = values[values > 0]
        return 0.0 if values.size == 0 else float(np.median(values))

    # Dark bottle-shape route.  Very tiny contours are hidden only to keep the
    # debug image readable; every contour near the detector's size range is
    # still shown even when it fails one or more checks.
    dark_min_display_area = max(15.0, float(cfg['min_area']) * 0.15)
    dark_contours = (contours(dark_mask)
                     if cfg.get('enable_dark_shape', True) else [])
    for contour in dark_contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        if area < dark_min_display_area or width < 3 or height < 8:
            continue

        contour_mask = np.zeros((height, width), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= x
        shifted[:, 0, 1] -= y
        cv2.drawContours(contour_mask, [shifted], -1, 255, cv2.FILLED)
        aspect = height / float(max(width, 1))
        bottom_ratio = (y + height) / float(max(roi_h, 1))
        fill_ratio = area / float(max(width * height, 1))
        mirrored = cv2.flip(contour_mask, 1)
        intersection = np.count_nonzero(
            (contour_mask > 0) & (mirrored > 0))
        union = np.count_nonzero((contour_mask > 0) | (mirrored > 0))
        symmetry = intersection / float(max(union, 1))
        widths = row_widths(contour_mask)
        top_end = max(1, int(height * 0.28))
        body_start = min(height - 1, int(height * 0.38))
        body_end = max(body_start + 1, int(height * 0.82))
        top_width = median_positive(widths[:top_end])
        body_width = median_positive(widths[body_start:body_end])
        shoulder_ratio = top_width / max(body_width, 1.0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        center_x = x + width * 0.5
        center_bonus = 1.0 - abs(center_x - roi_w * 0.5) / max(
            roi_w * 0.5, 1.0)
        aspect_score = math.exp(
            -abs(aspect - cfg['target_aspect']) /
            max(cfg['target_aspect'], 0.1))
        shoulder_score = max(0.0, min(
            1.0,
            (cfg['max_shoulder_ratio'] - shoulder_ratio) /
            max(cfg['max_shoulder_ratio'] -
                cfg['min_shoulder_ratio'], 0.01)))
        size_score = max(0.0, min(
            1.0, area / float(cfg['full_size_area'])))
        score = 100.0 * (
            0.25 * aspect_score + 0.22 * symmetry +
            0.20 * shoulder_score + 0.13 * solidity +
            0.10 * fill_ratio + 0.10 * center_bonus
        ) * (0.75 + 0.25 * size_score)

        checks = []
        check(checks, 'area', '%.0f' % area,
              '%.0f..%.0f' % (cfg['min_area'], cfg['max_area']),
              cfg['min_area'] <= area <= cfg['max_area'])
        check(checks, 'width', width,
              '%d..%d' % (cfg['min_width'], cfg['max_width']),
              cfg['min_width'] <= width <= cfg['max_width'])
        check(checks, 'height', height,
              '%d..%d' % (cfg['min_height'], cfg['max_height']),
              cfg['min_height'] <= height <= cfg['max_height'])
        check(checks, 'aspect', '%.2f' % aspect,
              '%.2f..%.2f' % (cfg['min_aspect'], cfg['max_aspect']),
              cfg['min_aspect'] <= aspect <= cfg['max_aspect'])
        check(checks, 'bottom', '%.2f' % bottom_ratio,
              '>=%.2f' % cfg['min_bottom_ratio'],
              bottom_ratio >= cfg['min_bottom_ratio'])
        check(checks, 'fill', '%.2f' % fill_ratio,
              '%.2f..%.2f' % (
                  cfg['min_fill_ratio'], cfg['max_fill_ratio']),
              cfg['min_fill_ratio'] <= fill_ratio <=
              cfg['max_fill_ratio'])
        check(checks, 'symmetry', '%.2f' % symmetry,
              '>=%.2f' % cfg['min_symmetry'],
              symmetry >= cfg['min_symmetry'])
        check(checks, 'shoulder', '%.2f' % shoulder_ratio,
              '%.2f..%.2f' % (
                  cfg['min_shoulder_ratio'], cfg['max_shoulder_ratio']),
              cfg['min_shoulder_ratio'] <= shoulder_ratio <=
              cfg['max_shoulder_ratio'])
        check(checks, 'solidity', '%.2f' % solidity,
              '>=%.2f' % cfg['min_solidity'],
              solidity >= cfg['min_solidity'])
        check(checks, 'score', '%.1f' % score,
              '>=%.1f' % cfg['min_score'], score >= cfg['min_score'])
        records.append({
            'method': 'dark',
            'bbox': (x1 + x, y1 + y, x1 + x + width, y1 + y + height),
            'score': float(score),
            'checks': checks,
            'accepted': all(item['passed'] for item in checks),
            'area': area,
        })

    # Red-cap anchor route.  A cap that passes its own color/shape checks but
    # has no plausible dark body is kept as an explicit rejected candidate.
    cap_min_display_area = max(
        2.0, cfg['cap_min_area_ratio'] * roi_area * 0.25)
    for cap_contour in contours(cap_mask):
        cap_area = float(cv2.contourArea(cap_contour))
        cap_x, cap_y, cap_w, cap_h = cv2.boundingRect(cap_contour)
        if cap_area < cap_min_display_area or cap_w <= 0 or cap_h <= 0:
            continue
        cap_area_ratio = cap_area / roi_area
        cap_aspect = cap_w / float(cap_h)
        cap_checks = []
        check(cap_checks, 'cap_area', '%.6f' % cap_area_ratio,
              '%.6f..%.3f' % (
                  cfg['cap_min_area_ratio'], cfg['cap_max_area_ratio']),
              cfg['cap_min_area_ratio'] <= cap_area_ratio <=
              cfg['cap_max_area_ratio'])
        check(cap_checks, 'cap_aspect', '%.2f' % cap_aspect,
              '%.2f..%.2f' % (
                  cfg['cap_min_aspect'], cfg['cap_max_aspect']),
              cfg['cap_min_aspect'] <= cap_aspect <=
              cfg['cap_max_aspect'])
        cap_bbox = (x1 + cap_x, y1 + cap_y,
                    x1 + cap_x + cap_w, y1 + cap_y + cap_h)
        if not all(item['passed'] for item in cap_checks):
            records.append({
                'method': 'cap', 'bbox': cap_bbox, 'score': 0.0,
                'checks': cap_checks, 'accepted': False, 'area': cap_area,
            })
            continue

        if cfg.get('cap_only_mode', False):
            cap_fill = cap_area / float(max(cap_w * cap_h, 1))
            cap_aspect_score = math.exp(-abs(cap_aspect - 1.0))
            cap_score = 100.0 * (
                0.65 * cap_aspect_score + 0.35 * cap_fill)
            records.append({
                'method': 'cap', 'bbox': cap_bbox,
                'score': float(cap_score), 'checks': cap_checks,
                'accepted': True, 'area': cap_area,
            })
            continue

        cap_center_x = cap_x + cap_w * 0.5
        search_half_w = int(max(cap_w * 2.4, roi_w * 0.025))
        search_x1 = max(0, int(cap_center_x) - search_half_w)
        search_x2 = min(roi_w, int(cap_center_x) + search_half_w + 1)
        search_y1 = min(roi_h, cap_y + cap_h)
        search_height = int(max(cap_h * 18.0, roi_h * 0.18))
        search_y2 = min(roi_h, search_y1 + search_height)
        local = dark_mask[search_y1:search_y2, search_x1:search_x2]
        bodies = contours(local) if local.size else []
        body_records = 0
        for body_contour in bodies:
            body_area = float(cv2.contourArea(body_contour))
            local_x, local_y, body_w, body_h = cv2.boundingRect(body_contour)
            if body_area < 10.0 or body_w <= 0 or body_h <= 0:
                continue
            body_records += 1
            body_x = search_x1 + local_x
            body_y = search_y1 + local_y
            body_center_x = body_x + body_w * 0.5
            body_to_cap_width = body_w / float(max(cap_w, 1))
            x_error = abs(body_center_x - cap_center_x)
            max_x_error = max(body_w * 0.30, cap_w * 1.0)
            gap = max(0, body_y - (cap_y + cap_h))
            max_gap = max(body_h * 0.22, cap_h * 3.0)
            box_x1 = min(cap_x, body_x)
            box_y1 = cap_y
            box_x2 = max(cap_x + cap_w, body_x + body_w)
            box_y2 = body_y + body_h
            aspect = (box_y2 - box_y1) / float(max(box_x2 - box_x1, 1))
            alignment = 1.0 - max(0.0, min(
                1.0, x_error / max((box_x2 - box_x1) * 0.5, 1.0)))
            aspect_score = math.exp(-abs(aspect - 2.6) / 2.6)
            gap_score = 1.0 - max(
                0.0, min(1.0, gap / max(max_gap, 1.0)))
            score = 100.0 * (
                0.50 * alignment + 0.30 * gap_score +
                0.20 * aspect_score)

            checks = list(cap_checks)
            min_body_area = max(
                cfg['cap_body_min_area'],
                cap_area * cfg['cap_body_area_gain'])
            check(checks, 'body_area', '%.0f' % body_area,
                  '>=%.0f' % min_body_area, body_area >= min_body_area)
            min_body_h = max(cfg['cap_body_min_height'], cap_h * 2.0)
            check(checks, 'body_height', body_h,
                  '>=%.0f' % min_body_h, body_h >= min_body_h)
            min_body_w = cfg['cap_body_min_width']
            check(checks, 'body_width', body_w,
                  '>=%d' % min_body_w, body_w >= min_body_w)
            check(checks, 'width_ratio', '%.2f' % body_to_cap_width,
                  '1.25..6.00', 1.25 <= body_to_cap_width <= 6.0)
            check(checks, 'x_align', '%.1f' % x_error,
                  '<=%.1f' % max_x_error, x_error <= max_x_error)
            check(checks, 'gap', '%.1f' % gap,
                  '<=%.1f' % max_gap, gap <= max_gap)
            check(checks, 'bottle_aspect', '%.2f' % aspect,
                  '%.2f..%.2f' % (
                      cfg['cap_bottle_min_aspect'],
                      cfg['cap_bottle_max_aspect']),
                  cfg['cap_bottle_min_aspect'] <= aspect <=
                  cfg['cap_bottle_max_aspect'])
            records.append({
                'method': 'cap',
                'bbox': (x1 + box_x1, y1 + box_y1,
                         x1 + box_x2, y1 + box_y2),
                'score': float(score),
                'checks': checks,
                'accepted': all(item['passed'] for item in checks),
                'area': body_area,
            })
        if body_records == 0:
            checks = list(cap_checks)
            check(checks, 'dark_body', 'none', 'required', False)
            records.append({
                'method': 'cap', 'bbox': cap_bbox, 'score': 0.0,
                'checks': checks, 'accepted': False, 'area': cap_area,
            })

    return records, dark_full, cap_full


def _cola_debug_view(frame, cfg, selected):
    records, dark_mask, cap_mask = _cola_debug_candidates(frame, cfg)
    view = frame.copy()
    if not cfg.get('cap_only_mode', False):
        dark_tint = np.zeros_like(view)
        dark_tint[:, :, 0] = dark_mask
        view = cv2.addWeighted(view, 1.0, dark_tint, 0.22, 0.0)
    cap_tint = np.zeros_like(view)
    cap_tint[:, :, 2] = cap_mask
    view = cv2.addWeighted(view, 1.0, cap_tint, 0.28, 0.0)

    # Prefer accepted and nearly-passing candidates, while avoiding hundreds
    # of labels from insignificant color speckles.
    records.sort(key=lambda item: (
        item['accepted'],
        sum(check_item['passed'] for check_item in item['checks']),
        item['score'], item['area']), reverse=True)
    selected_bbox = tuple(selected.bbox_img) if selected is not None else None
    selected_method = (str(selected.extra.get('method', ''))
                       if selected is not None else '')
    visible = records[:12]
    if selected_bbox is not None:
        selected_record = next((
            item for item in records
            if tuple(int(value) for value in item['bbox']) == selected_bbox
            and (selected_method.startswith('cap') ==
                 (item['method'] == 'cap'))
        ), None)
        if selected_record is not None and selected_record not in visible:
            visible = records[:11] + [selected_record]
    detail = None
    for index, item in enumerate(visible, 1):
        item['id'] = ('C' if item['method'] == 'cap' else 'D') + str(index)
        bbox = tuple(int(value) for value in item['bbox'])
        is_selected = (
            selected_bbox == bbox and
            (selected_method.startswith('cap') == (item['method'] == 'cap')))
        if is_selected:
            detail = item
        color = (0, 220, 0) if item['accepted'] else (0, 0, 255)
        thickness = 3 if is_selected else 1
        x1, y1, x2, y2 = bbox
        cv2.rectangle(view, (x1, y1), (x2, y2), color, thickness)
        failed = [entry['name'] for entry in item['checks']
                  if not entry['passed']]
        summary = 'PASS' if not failed else 'FAIL ' + ','.join(failed[:2])
        _put_label(view, '%s %s %.1f' % (
            item['id'], summary, item['score']), (x1, y1 - 5), color, 0.38)

    if detail is None and visible:
        detail = visible[0]
    panel_width = min(max(290, int(view.shape[1] * 0.48)), view.shape[1] - 10)
    panel_height = min(view.shape[0] - 10, 32 + 15 * 14)
    overlay = view.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_width, 5 + panel_height),
                  (8, 12, 18), -1)
    view = cv2.addWeighted(overlay, 0.82, view, 0.18, 0.0)
    if cfg.get('cap_only_mode', False):
        mode_text = 'RED CAP ONLY'
    elif cfg.get('enable_dark_shape', True):
        mode_text = 'CAP ROUTE + DARK FALLBACK'
    else:
        mode_text = 'CAP + DARK BODY'
    _put_label(view, 'GREEN=PASS  RED=FAIL  %s' % mode_text,
               (12, 23), (255, 255, 255), 0.40)
    if detail is None:
        _put_label(view, 'NO VISIBLE COLA CANDIDATE',
                   (12, 48), (0, 165, 255), 0.55)
    else:
        title_color = (0, 220, 0) if detail['accepted'] else (0, 0, 255)
        selection_suffix = ''
        if selected is not None:
            centerline_distance = selected.extra.get(
                'selection_centerline_distance_px')
            if centerline_distance is not None:
                area_bonus = float(selected.extra.get(
                    'selection_area_bonus_px', 0.0))
                selection_suffix = (
                    ' centerline_dist=%.1fpx area_bonus=%.1fpx' % (
                        float(centerline_distance), area_bonus))
        _put_label(view, '%s %s score=%.1f%s' % (
            detail.get('id', '?'),
            'PASS' if detail['accepted'] else 'FAIL', detail['score'],
            selection_suffix),
            (12, 46), title_color, 0.50)
        line_y = 66
        for entry in detail['checks'][:13]:
            mark = '+' if entry['passed'] else '-'
            color = (80, 230, 80) if entry['passed'] else (80, 80, 255)
            text = '%s %-14s %s  rule:%s' % (
                mark, entry['name'], entry['value'], entry['rule'])
            _put_label(view, text, (12, line_y), color, 0.34)
            line_y += 15
    return view


def _mask_tile(mask, label, size):
    tile = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    _put_label(tile, label, (8, 22), (0, 255, 255), 0.52)
    return tile


def _mask_mosaic(frame, configs):
    bar = _full_hsv_mask(
        frame, configs['bar'],
        (configs['bar']['open_kernel'],) * 2,
        (configs['bar']['close_kernel_h'], configs['bar']['close_kernel_w']))
    obstacle = _full_hsv_mask(
        frame, configs['obstacle'],
        (configs['obstacle']['open_kernel'],) * 2,
        (configs['obstacle']['close_kernel'],) * 2)
    dash = _roi_mask(
        frame, configs['yellow'],
        lambda roi: _full_hsv_mask(
            roi, configs['yellow'],
            (configs['yellow']['open_kernel'],) * 2,
            (configs['yellow']['dash_close_kernel_h'],
             configs['yellow']['dash_close_kernel_w'])))
    final = _roi_mask(
        frame, configs['final_yellow'],
        lambda roi: _full_hsv_mask(
            roi, configs['final_yellow'],
            (configs['final_yellow']['open_kernel'],) * 2,
            (configs['final_yellow']['close_kernel_h'],
             configs['final_yellow']['close_kernel_w'])))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    football_black = cv2.inRange(
        hsv, np.array([0, 0, 0], dtype=np.uint8),
        np.array([179, 255, 104], dtype=np.uint8))
    football_white = cv2.inRange(
        hsv, np.array([0, 0, 111], dtype=np.uint8),
        np.array([179, 104, 255], dtype=np.uint8))
    football = cv2.bitwise_or(football_black, football_white)
    cola_dark, cola_cap = _cola_masks(frame, configs['cola'])
    blank = np.zeros_like(football)
    masks = [bar, obstacle, dash, final, football, cola_dark, cola_cap, blank]
    labels = ['BAR', 'OBSTACLE', 'DASH', 'FINAL YELLOW',
              'FOOTBALL B/W', 'COLA DARK', 'COLA CAP', '']
    h, w = frame.shape[:2]
    tile_size = (max(1, w // 4), max(1, h // 2))
    tiles = [_mask_tile(mask, label, tile_size)
             for mask, label in zip(masks, labels)]
    rows = [cv2.hconcat(tiles[index:index + 4])
            for index in range(0, len(tiles), 4)]
    return cv2.vconcat(rows)


def _depth_visual(depth_m, output_shape):
    h, w = output_shape[:2]
    if depth_m is None:
        image = np.zeros((h, w, 3), np.uint8)
        _put_label(image, 'NO DEPTH FRAME', (20, 35), (0, 0, 255), 0.8)
        return image
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.clip(depth_m, 0.2, 3.0)
    normalized = ((clipped - 0.2) / 2.8 * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    if colored.shape[:2] != (h, w):
        colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_NEAREST)
    _put_label(colored, 'DEPTH 0.2m..3.0m', (10, 24), (255, 255, 255), 0.55)
    return colored


class PreviewState(object):
    def __init__(self, stale_after):
        self.lock = threading.Lock()
        self.stale_after = float(stale_after)
        self.rgb_msg = None
        self.depth_msg = None
        self.ai_msg = None
        self.rgb_seq = 0
        self.depth_seq = 0
        self.ai_seq = 0
        self.processed_rgb_seq = 0
        self.rgb_received = 0
        self.depth_received = 0
        self.ai_received = 0
        self.dropped = 0
        self.last_rgb = 0.0
        self.last_depth = 0.0
        self.last_ai = 0.0
        self.last_process = 0.0
        self.processing_started = 0.0
        self.processing_step = 'idle'
        self.last_process_duration = None
        self.process_times = collections.deque(maxlen=60)
        self.frames = {key: None for key in STREAMS}
        self.frame_seq = 0
        self.detections = []
        self.warning = ''
        self.last_error = ''
        self.rgb_shape = None
        self.depth_shape = None
        self.ai_shape = None
        self.rgb_publishers = 0
        self.depth_publishers = 0
        self.ai_publishers = 0

    def push_rgb(self, msg):
        with self.lock:
            if self.rgb_msg is not None and self.rgb_seq > self.processed_rgb_seq:
                self.dropped += 1
            self.rgb_msg = msg
            self.rgb_seq += 1
            self.rgb_received += 1
            self.last_rgb = time.monotonic()

    def push_depth(self, msg):
        with self.lock:
            self.depth_msg = msg
            self.depth_seq += 1
            self.depth_received += 1
            self.last_depth = time.monotonic()

    def push_ai(self, msg):
        with self.lock:
            self.ai_msg = msg
            self.ai_seq += 1
            self.ai_received += 1
            self.last_ai = time.monotonic()

    def claim(self):
        with self.lock:
            if self.rgb_msg is None or self.rgb_seq <= self.processed_rgb_seq:
                return None
            self.processed_rgb_seq = self.rgb_seq
            ai_age = (max(0.0, time.monotonic() - self.last_ai)
                      if self.last_ai else None)
            return (self.rgb_seq, self.rgb_msg, self.depth_msg, self.ai_msg,
                    self.ai_seq, ai_age)

    def begin_processing(self):
        with self.lock:
            self.processing_started = time.monotonic()
            self.processing_step = 'decode'

    def set_processing_step(self, step):
        with self.lock:
            self.processing_step = str(step)

    def commit(self, frames, detections, warning, rgb_shape, depth_shape,
               ai_shape):
        now = time.monotonic()
        with self.lock:
            self.frames = dict(frames)
            self.frame_seq += 1
            self.detections = list(detections)
            self.warning = str(warning or '')
            self.rgb_shape = rgb_shape
            self.depth_shape = depth_shape
            self.ai_shape = ai_shape
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

    def set_publishers(self, rgb_count, depth_count, ai_count):
        with self.lock:
            self.rgb_publishers = int(rgb_count)
            self.depth_publishers = int(depth_count)
            self.ai_publishers = int(ai_count)

    def frame(self, key, after):
        with self.lock:
            if key not in self.frames or self.frames[key] is None:
                return None
            if after is not None and after >= self.frame_seq:
                return self.frame_seq, None, 0.0
            age = max(0.0, time.monotonic() - self.last_process)
            return self.frame_seq, self.frames[key], age

    def status(self, node):
        now = time.monotonic()
        with self.lock:
            rgb_age = max(0.0, now - self.last_rgb) if self.last_rgb else None
            ai_age = max(0.0, now - self.last_ai) if self.last_ai else None
            process_age = (max(0.0, now - self.processing_started)
                           if self.processing_started else None)
            warning = self.warning
            if self.last_error:
                health = 'error'
            elif (process_age is not None and
                  process_age > node.args.processing_timeout):
                health = 'error'
                warning = ('处理线程停在 %s 已 %.1f 秒；相机回调仍可继续，'
                           '请查看终端中的 PREVIEW_STALL 日志。') % (
                               self.processing_step, process_age)
            elif rgb_age is None or rgb_age > self.stale_after:
                health = 'waiting'
            else:
                health = 'live'
            fps = None
            if len(self.process_times) >= 2:
                elapsed = self.process_times[-1] - self.process_times[0]
                if elapsed > 0:
                    fps = (len(self.process_times) - 1) / elapsed
            return {
                'health': health,
                'platform': node.platform,
                'rgb_topic': node.rgb_topic,
                'depth_topic': node.depth_topic,
                'ai_topic': node.ai_topic,
                'rgb_publishers': self.rgb_publishers,
                'depth_publishers': self.depth_publishers,
                'ai_publishers': self.ai_publishers,
                'rgb_received': self.rgb_received,
                'depth_received': self.depth_received,
                'ai_received': self.ai_received,
                'rgb_age_s': round(rgb_age, 3) if rgb_age is not None else None,
                'ai_age_s': round(ai_age, 3) if ai_age is not None else None,
                'rgb_shape': self.rgb_shape,
                'depth_shape': self.depth_shape,
                'ai_shape': self.ai_shape,
                'detections': list(self.detections),
                'warning': warning,
                'last_error': self.last_error,
                'dropped': self.dropped,
                'process_fps': round(fps, 2) if fps is not None else None,
                'process_duration_s': (round(self.last_process_duration, 3)
                                       if self.last_process_duration is not None
                                       else None),
                'processing_step': self.processing_step,
                'processing_age_s': (round(process_age, 3)
                                     if process_age is not None else None),
                'bar_hsv': '%d..%d / S%d..%d / V%d..%d' % (
                    node.configs['bar']['h_min'], node.configs['bar']['h_max'],
                    node.configs['bar']['s_min'], node.configs['bar']['s_max'],
                    node.configs['bar']['v_min'], node.configs['bar']['v_max']),
            }


class Stage4VisionPreviewNode(Node):
    def __init__(self, args):
        Node.__init__(self, 'stage4_vision_preview')
        self.args = args
        self.bridge = CvBridge()
        cv2.setNumThreads(max(1, int(args.opencv_threads)))
        self.declare_parameter('platform', args.platform)
        self.platform = str(self.get_parameter('platform').value).strip().lower()
        if self.platform not in ('sim', 'real'):
            raise ValueError("platform must be 'sim' or 'real'")
        self.declare_parameter('rgb_topic', args.rgb_topic or '')
        self.declare_parameter('depth_topic', args.depth_topic or '')
        self.declare_parameter('ai_camera_topic', args.ai_topic or '')

        defaults = collections.OrderedDict([
            ('bar', _bar_defaults(self.platform)),
            ('obstacle', OBSTACLE_DEFAULTS),
            ('yellow', YELLOW_DEFAULTS),
            ('final_yellow', FINAL_YELLOW_DEFAULTS),
            ('blue_ball', BLUE_BALL_DEFAULTS),
            ('white_ball', WHITE_BALL_DEFAULTS),
            ('cola', COLA_DEFAULTS),
        ])
        defaults['cola'] = dict(defaults['cola'])
        defaults['obstacle'] = dict(defaults['obstacle'])
        defaults['obstacle']['use_depth_filter'] = self.platform != 'real'
        defaults['cola']['cap_only_mode'] = self.platform == 'real'
        defaults['cola']['enable_dark_shape'] = self.platform != 'real'
        if self.platform == 'real':
            defaults['obstacle'].update({
                'min_area': 7000,
                'max_area': 40000,
                'min_aspect_ratio': 0.7,
                'max_aspect_ratio': 1.3,
                'min_bottom_y_ratio_in_roi': 0.5,
            })
            defaults['cola'].update({
                'dark_v_max': 145,
                'dark_s_min': 20,
                'very_dark_v_max': 70,
                'close_kernel': 9,
            })
        for prefix, values in defaults.items():
            _declare_config(self, prefix, values)
        self.configs = collections.OrderedDict(
            (prefix, _read_config(self, prefix, values))
            for prefix, values in defaults.items())
        for name, value in RGB_TRIGGER_DEFAULTS.items():
            self.declare_parameter(name, value)
        self.rgb_trigger_cfg = {
            name: float(self.get_parameter(name).value)
            for name in RGB_TRIGGER_DEFAULTS
        }
        for name, value in BASKETBALL_AI_DEFAULTS.items():
            self.declare_parameter(name, value)
        self.basketball_ai_cfg = {
            name: self.get_parameter(name).value
            for name in BASKETBALL_AI_DEFAULTS
        }
        for name in self.basketball_ai_cfg:
            if name == 'basketball_top_trigger_confirm_frames':
                self.basketball_ai_cfg[name] = int(
                    self.basketball_ai_cfg[name])
            else:
                self.basketball_ai_cfg[name] = float(
                    self.basketball_ai_cfg[name])
        ai_detector_cfg = self.configs['blue_ball']
        ai_detector_cfg.update({
            'roi_x_ratio_min': self.basketball_ai_cfg[
                'basketball_ai_roi_x_min_ratio'],
            'roi_x_ratio_max': self.basketball_ai_cfg[
                'basketball_ai_roi_x_max_ratio'],
            'roi_y_ratio_min': self.basketball_ai_cfg[
                'basketball_ai_roi_y_min_ratio'],
            'roi_y_ratio_max': self.basketball_ai_cfg[
                'basketball_ai_roi_y_max_ratio'],
        })

        self.bar_detector = BarColorDetector(self.configs['bar'])
        self.obstacle_detector = ObstacleBlueDetector(self.configs['obstacle'])
        self.dashed_detector = YellowDashedLineDetector(self.configs['yellow'])
        self.final_detector = YellowHorizontalLineDetector(
            self.configs['final_yellow'])
        self.blue_ball_detector = BallDetector(
            self.configs['blue_ball'], 'blue_ball')
        self.white_ball_detector = FootballDetector()
        self.cola_detector = ColaDetector(self.configs['cola'])

        self.state = PreviewState(args.stale_after)
        self.rgb_topic = ''
        self.depth_topic = ''
        self.ai_topic = ''
        self.basketball_trigger_count = 0
        self.last_basketball_ai_seq = 0
        self._subscriptions = []
        self._publisher_timer = self.create_timer(1.0, self._update_publishers)
        self._watchdog_timer = self.create_timer(1.0, self._check_worker)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop,
                                        name='stage4-preview-worker')
        self._worker.daemon = True

    def attach(self, rgb_topic, depth_topic, ai_topic):
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic)
        self.ai_topic = str(ai_topic)
        qos = _image_qos()
        self._subscriptions.append(self.create_subscription(
            Image, self.rgb_topic, self.state.push_rgb, qos))
        self._subscriptions.append(self.create_subscription(
            Image, self.depth_topic, self.state.push_depth, qos))
        self._subscriptions.append(self.create_subscription(
            Image, self.ai_topic, self.state.push_ai, qos))
        self._worker.start()
        self._update_publishers()
        self.get_logger().info('RGB topic: %s' % self.rgb_topic)
        self.get_logger().info('Depth topic: %s' % self.depth_topic)
        self.get_logger().info('AI camera topic: %s' % self.ai_topic)

    def close(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def _update_publishers(self):
        if not self.rgb_topic:
            return
        try:
            rgb_count = self.count_publishers(self.rgb_topic)
            depth_count = self.count_publishers(self.depth_topic)
            ai_count = self.count_publishers(self.ai_topic)
        except Exception:
            rgb_count, depth_count, ai_count = 0, 0, 0
        self.state.set_publishers(rgb_count, depth_count, ai_count)

    def _check_worker(self):
        age, step = self.state.processing_snapshot()
        if age > self.args.processing_timeout:
            self.get_logger().error(
                '[PREVIEW_STALL] processing stuck at %s for %.1fs' % (
                    step, age),
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
            _, rgb_msg, depth_msg, ai_msg, ai_seq, ai_age_sec = item
            self.state.begin_processing()
            try:
                self._process(
                    rgb_msg, depth_msg, ai_msg, ai_seq, ai_age_sec)
            except Exception as error:
                self.state.error(error)
                self.get_logger().warning('preview processing failed: %s' % error)

    def _process(self, rgb_msg, depth_msg, ai_msg, ai_seq, ai_age_sec):
        self.state.set_processing_step('decode')
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        ai_frame = None
        if ai_msg is not None:
            ai_frame = self.bridge.imgmsg_to_cv2(
                ai_msg, desired_encoding='bgr8')
        depth = None
        if depth_msg is not None:
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        depth_m = _depth_meters(depth)
        warning = ''
        same_shape = depth_m is not None and depth_m.shape[:2] == frame.shape[:2]
        obstacle_uses_depth = bool(
            self.configs['obstacle'].get('use_depth_filter', True))
        if depth_m is None:
            if obstacle_uses_depth:
                warning = '尚未收到深度图：仿真障碍物深度过滤暂不可用。'
        elif not same_shape:
            warning = ('RGB 与 Depth 尺寸不同（%s / %s）。'
                       '真机障碍物为纯 RGB 检测；深度仅用于辅助显示。') % (
                           frame.shape[:2], depth_m.shape[:2])

        self.state.set_processing_step('bar')
        bar = self.bar_detector.detect(frame, depth)
        self.state.set_processing_step('obstacle')
        if not obstacle_uses_depth or same_shape:
            obstacle_result = self.obstacle_detector.detect(frame, depth)
            obstacles = obstacle_result['candidates']
            obstacles.sort(key=lambda det: (
                int(det.bbox_img[3]),
                float(det.extra.get('area', 0.0)),
            ), reverse=True)
            obstacle_debug = obstacle_result['debug_infos']
        else:
            obstacles = []
            obstacle_debug = []
        self.state.set_processing_step('yellow')
        dashed = self.dashed_detector.detect_top_dashed_lines(frame)
        final = self.final_detector.detect(frame)
        self.state.set_processing_step('targets')
        targets = []
        cola_result = None
        ai_fresh = (ai_frame is not None and ai_age_sec is not None and
                    ai_age_sec <= self.basketball_ai_cfg[
                        'basketball_ai_max_age_s'])
        blue_ball = (self.blue_ball_detector.detect(ai_frame)
                     if ai_fresh else None)
        if ai_seq != self.last_basketball_ai_seq:
            self.last_basketball_ai_seq = ai_seq
            top_ratio = None
            if blue_ball is not None:
                top_ratio = (float(blue_ball.bbox_img[1]) /
                             float(max(ai_frame.shape[0], 1)))
            if (top_ratio is not None and top_ratio <=
                    self.basketball_ai_cfg[
                        'basketball_top_trigger_y_ratio']):
                self.basketball_trigger_count += 1
            else:
                self.basketball_trigger_count = 0
        elif not ai_fresh:
            self.basketball_trigger_count = 0

        def football_depth_at(x, y):
            value = _depth_at(depth_m, (x, y), frame.shape)
            return -1.0 if value is None else value

        white_ball = self.white_ball_detector.detect(
            frame, depth_at=football_depth_at)
        if white_ball is not None:
            targets.append(white_ball)

        image_center_x = frame.shape[1] * 0.5
        max_cap_area_ratio = max(
            float(self.configs['cola']['cap_max_area_ratio']), 1e-9)
        max_area_bonus_px = frame.shape[1] * 0.05

        def nearest_cola_key(candidate):
            centerline_distance_px = abs(
                float(candidate['center'][0]) - image_center_x)
            cap_area_ratio = float(candidate.get('cap_area_ratio', 0.0))
            area_bonus_px = max_area_bonus_px * min(
                cap_area_ratio / max_cap_area_ratio, 1.0)
            selection_cost = centerline_distance_px - area_bonus_px
            candidate['selection_centerline_distance_px'] = (
                centerline_distance_px)
            candidate['selection_area_bonus_px'] = area_bonus_px
            return (selection_cost, centerline_distance_px,
                    -float(candidate['score']))

        cola_result = self.cola_detector.detect(
            frame, candidate_key=nearest_cola_key)
        if cola_result is not None:
            targets.append(cola_result)

        self.state.set_processing_step('cola_debug')
        cola_debug_view = _cola_debug_view(
            frame, self.configs['cola'], cola_result)
        bar_obstacle_debug_view = _bar_obstacle_debug_view(
            frame,
            depth,
            self.bar_detector,
            self.configs['bar'],
            bar,
            obstacle_debug,
            self.configs['obstacle'],
        )
        if ai_frame is None:
            ai_source_view = np.zeros_like(frame)
            _put_label(ai_source_view, 'WAITING FOR AI CAMERA',
                       (20, 35), (0, 0, 255), 0.70)
        else:
            ai_source_view = ai_frame
        basketball_ai_view = _basketball_ai_view(
            ai_source_view,
            blue_ball,
            self.configs['blue_ball'],
            self.basketball_ai_cfg,
            self.ai_topic,
            ai_age_sec,
            self.basketball_trigger_count,
        )

        bar_obstacle_view = frame.copy()
        targets_view = frame.copy()
        yellow_view = frame.copy()
        h, w = frame.shape[:2]
        for view in (bar_obstacle_view, targets_view, yellow_view):
            cv2.line(view, (w // 2, 0), (w // 2, h - 1),
                     (0, 255, 0), 1)
        _roi_box(targets_view, self.configs['cola'], (0, 0, 255), 'COLA')
        _roi_box(yellow_view, self.configs['yellow'], (0, 165, 255), 'DASH')
        _roi_box(yellow_view, self.configs['final_yellow'],
                 (0, 255, 255), 'FINAL')
        records = []

        def record(view, det, label, color):
            if det is None:
                return
            center_depth = _depth_at(depth_m, det.center_img, frame.shape)
            display_label = label
            if det.det_type == 'bar':
                top_px = int(det.bbox_img[1])
                top_ratio = float(top_px) / float(max(frame.shape[0], 1))
                display_label += ' top=%dpx/%.3f' % (top_px, top_ratio)
            elif det.det_type == 'blue_obstacle':
                bottom_px = int(det.bbox_img[3])
                bottom_ratio = float(bottom_px) / float(max(frame.shape[0], 1))
                display_label += ' bottom=%dpx/%.3f' % (
                    bottom_px, bottom_ratio)
            elif det.det_type == 'white_ball':
                display_label += ' area=%.0fpx2/%.4f r=%.1fpx' % (
                    float(det.extra.get('area', 0.0)),
                    float(det.extra.get('area_ratio', 0.0)),
                    float(det.extra.get('radius', 0.0)))
            elif det.det_type == 'cola':
                display_label += ' cap=%.0fpx2/%.5f' % (
                    float(det.extra.get('area', 0.0)),
                    float(det.extra.get('cap_area_ratio', 0.0)))
            control_depth = center_depth if self.platform != 'real' else None
            _draw_detection(view, det, color, display_label, control_depth)
            item = {
                'type': det.det_type,
                'center': [int(det.center_img[0]), int(det.center_img[1])],
                'bbox': [int(value) for value in det.bbox_img],
                'score': round(float(det.score), 2),
                'depth_m': round(center_depth, 3) if center_depth is not None else None,
            }
            method = det.extra.get('method')
            if method:
                item['method'] = str(method)
            if det.det_type in ('bar', 'blue_obstacle'):
                edge_y_ratio = (
                    float(det.bbox_img[1]) / float(max(frame.shape[0], 1))
                    if det.det_type == 'bar'
                    else float(det.bbox_img[3]) / float(max(frame.shape[0], 1)))
                if det.det_type == 'bar':
                    item['top_y_px'] = int(det.bbox_img[1])
                    item['top_y_ratio'] = round(float(edge_y_ratio), 4)
                else:
                    item['bottom_y_px'] = int(det.bbox_img[3])
                    item['bottom_y_ratio'] = round(float(edge_y_ratio), 4)
            elif det.det_type == 'white_ball':
                item['area_px'] = round(
                    float(det.extra.get('area', 0.0)), 1)
                item['area_ratio'] = round(
                    float(det.extra.get('area_ratio', 0.0)), 6)
                item['radius_px'] = round(
                    float(det.extra.get('radius', 0.0)), 2)
            elif det.det_type == 'cola':
                item['cap_area_px'] = round(
                    float(det.extra.get('area', 0.0)), 1)
                item['cap_area_ratio'] = round(
                    float(det.extra.get('cap_area_ratio', 0.0)), 6)
            records.append(item)

        record(bar_obstacle_view, bar, 'BAR', (0, 0, 255))
        for index, det in enumerate(obstacles):
            record(bar_obstacle_view, det, 'OBS%d' % index, (255, 0, 0))
        for index, det in enumerate(dashed):
            record(yellow_view, det, 'DASH%d' % index, (0, 165, 255))
            centers = det.extra.get('group_centers', [])
            for center in centers:
                cv2.circle(yellow_view, (int(center[0]), int(center[1])),
                           3, (0, 165, 255), -1)
        record(yellow_view, final, 'FINAL YELLOW', (0, 255, 255))
        for det in targets:
            if det.det_type == 'white_ball':
                color = (255, 255, 255)
            else:
                color = (0, 0, 255)
            label = det.det_type.upper()
            if det.det_type == 'cola':
                label += ' ' + str(det.extra.get('method', 'unknown'))
            record(targets_view, det, label, color)
        if blue_ball is not None:
            ai_top_ratio = (float(blue_ball.bbox_img[1]) /
                            float(max(ai_frame.shape[0], 1)))
            records.append({
                'type': 'blue_ball',
                'source': 'ai_camera',
                'center': [int(blue_ball.center_img[0]),
                           int(blue_ball.center_img[1])],
                'bbox': [int(value) for value in blue_ball.bbox_img],
                'score': round(float(blue_ball.score), 2),
                'depth_m': None,
                'top_y_px': int(blue_ball.bbox_img[1]),
                'top_y_ratio': round(ai_top_ratio, 4),
                'area_px': round(float(blue_ball.extra.get('area', 0.0)), 1),
                'area_ratio': round(
                    float(blue_ball.extra.get('area_ratio', 0.0)), 6),
                'radius_px': round(
                    float(blue_ball.extra.get('radius', 0.0)), 2),
            })

        bar_obstacle_view = _bar_obstacle_trigger_view(
            bar_obstacle_view, bar, obstacles, self.rgb_trigger_cfg)
        targets_view = _target_trigger_view(
            targets_view, targets, self.rgb_trigger_cfg)

        self.state.set_processing_step('encode')
        frames = {
            'bar_obstacle': _encode_jpeg(
                bar_obstacle_view, self.args.jpeg_quality),
            'bar_obstacle_debug': _encode_jpeg(
                bar_obstacle_debug_view, self.args.jpeg_quality),
            'targets': _encode_jpeg(targets_view, self.args.jpeg_quality),
            'basketball_ai': _encode_jpeg(
                basketball_ai_view, self.args.jpeg_quality),
            'cola_debug': _encode_jpeg(
                cola_debug_view, self.args.jpeg_quality),
            'yellow': _encode_jpeg(yellow_view, self.args.jpeg_quality),
        }
        rgb_shape = '%dx%d %s' % (rgb_msg.width, rgb_msg.height, rgb_msg.encoding)
        depth_shape = None
        if depth_msg is not None:
            depth_shape = '%dx%d %s' % (
                depth_msg.width, depth_msg.height, depth_msg.encoding)
        ai_shape = None
        if ai_msg is not None:
            ai_shape = '%dx%d %s' % (
                ai_msg.width, ai_msg.height, ai_msg.encoding)
        self.state.commit(
            frames, records, warning, rgb_shape, depth_shape, ai_shape)


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
        server_version = 'CyberDogStage4Vision/1.0'

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
                self.send_payload(200, 'application/json; charset=utf-8', payload)
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
        description='Read-only Stage-4 RGB/depth detection web preview')
    parser.add_argument('--platform', choices=('sim', 'real'), default='real')
    parser.add_argument('--dog-ns', default='auto')
    parser.add_argument('--rgb-topic')
    parser.add_argument('--depth-topic')
    parser.add_argument('--ai-topic')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8084)
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
        rgb_default = _join_topic(namespace, 'image_rgb')
        depth_default = _join_topic(namespace, 'camera/depth/image_rect_raw')
        ai_default = '/mi_desktop_48_b0_2d_7b_00_e2/image'
    else:
        rgb_default = '/rgb_camera/rgb_camera/image_raw'
        depth_default = '/d435/depth/d435_depth/depth/image_raw'
        ai_default = rgb_default
    rgb_parameter = str(node.get_parameter('rgb_topic').value).strip()
    depth_parameter = str(node.get_parameter('depth_topic').value).strip()
    ai_parameter = str(node.get_parameter('ai_camera_topic').value).strip()
    return (
        args.rgb_topic or rgb_parameter or rgb_default,
        args.depth_topic or depth_parameter or depth_default,
        args.ai_topic or ai_parameter or ai_default,
    )


def main(argv=None):
    args, ros_args = _parse_args(argv)
    rclpy.init(args=ros_args)
    node = Stage4VisionPreviewNode(args)
    server = None
    server_thread = None
    try:
        rgb_topic, depth_topic, ai_topic = _resolve_topics(node, args)
        node.attach(rgb_topic, depth_topic, ai_topic)
        page = _build_page(args.client_fps)
        server = _ThreadedHTTPServer(
            (args.bind, args.port), _make_handler(node, page))
        server_thread = threading.Thread(target=server.serve_forever,
                                         name='stage4-preview-http')
        server_thread.daemon = True
        server_thread.start()
        node.get_logger().info(
            'Stage4 read-only preview: http://%s:%d/ (max %.1f FPS)' % (
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
