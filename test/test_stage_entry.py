#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage_entry 与六个赛段入口表的单元测试。

只导入纯模块，因此可以在没有 ROS 的主机上直接 pytest。赛段节点自己的入口表
定义在 stageN_node.py 里（会 import ROS），所以这里重建等价的表结构，重点是
锁住解析规则；表本身的自洽性（起点/状态/重名）由 StageEntryTable 构造时校验。
"""

import pytest

from control_node.stage_entry import (
    SOURCE_DEFAULT,
    SOURCE_ENTRY_POINT,
    SOURCE_FALLBACK,
    SOURCE_STATE,
    EntryPoint,
    StageEntryTable,
    is_default_request,
)


def make_table():
    states = ('P5_SET_BODY_NORMAL', 'P5_STEP_UP', 'P5_UP_SLOPE', 'P5_TURN_1')
    return StageEntryTable(5, 'P5_SET_BODY_NORMAL', states, (
        EntryPoint('start', 'P5_SET_BODY_NORMAL', '完整第五赛段'),
        EntryPoint('step_up', 'P5_STEP_UP'),
        EntryPoint('ramp', 'P5_UP_SLOPE', '上坡段', requires=('已经站在坡道上',)),
        EntryPoint('corner_2', 'P5_TURN_1'),
    ))


# ----------------------------------------------------------------------
# is_default_request
# ----------------------------------------------------------------------
@pytest.mark.parametrize('value', ['', '  ', 'default', 'DEFAULT', 'none',
                                   'auto', None])
def test_default_aliases(value):
    assert is_default_request(value) is True


@pytest.mark.parametrize('value', ['ramp', 'P5_UP_SLOPE', 'defaults'])
def test_non_default_requests(value):
    assert is_default_request(value) is False


# ----------------------------------------------------------------------
# resolve
# ----------------------------------------------------------------------
def test_default_request_uses_start_state():
    resolution = make_table().resolve('default')
    assert resolution.state == 'P5_SET_BODY_NORMAL'
    assert resolution.source == SOURCE_DEFAULT
    assert resolution.ok is True
    assert resolution.is_default is True


def test_empty_request_uses_start_state():
    assert make_table().resolve('').state == 'P5_SET_BODY_NORMAL'
    assert make_table().resolve(None).state == 'P5_SET_BODY_NORMAL'


def test_named_entry_point():
    resolution = make_table().resolve('ramp')
    assert resolution.state == 'P5_UP_SLOPE'
    assert resolution.source == SOURCE_ENTRY_POINT
    assert resolution.ok is True
    assert resolution.is_default is False
    assert resolution.requires == ('已经站在坡道上',)


@pytest.mark.parametrize('value', ['RAMP', ' Ramp ', 'ramp'])
def test_entry_name_matching_is_forgiving(value):
    assert make_table().resolve(value).state == 'P5_UP_SLOPE'


def test_hyphen_is_treated_as_underscore():
    assert make_table().resolve('corner-2').state == 'P5_TURN_1'


def test_raw_state_name_is_accepted():
    resolution = make_table().resolve('P5_TURN_1')
    assert resolution.state == 'P5_TURN_1'
    assert resolution.source == SOURCE_STATE
    assert resolution.ok is True
    # 该状态恰好也有具名入口，解析结果要把它带上，前提提示才不会丢。
    assert resolution.entry_point is not None


def test_raw_state_name_is_case_insensitive():
    assert make_table().resolve('p5_turn_1').state == 'P5_TURN_1'


def test_unknown_request_falls_back_without_raising():
    resolution = make_table().resolve('rmap')
    assert resolution.ok is False
    assert resolution.source == SOURCE_FALLBACK
    # 回退到正常起点：写错调试参数只是跑了一次正常流程，不是节点起不来。
    assert resolution.state == 'P5_SET_BODY_NORMAL'
    assert 'rmap' in resolution.message


def test_resolution_to_dict_is_serializable():
    data = make_table().resolve('ramp').to_dict()
    assert data['state'] == 'P5_UP_SLOPE'
    assert data['entry_point'] == 'ramp'
    assert data['ok'] is True


# ----------------------------------------------------------------------
# 表结构校验：这些是编码错误，要在没有机器人的情况下就抓住
# ----------------------------------------------------------------------
def test_start_state_must_be_a_declared_state():
    with pytest.raises(ValueError):
        StageEntryTable(1, 'NOPE', ('A', 'B'))


def test_entry_point_must_target_a_declared_state():
    with pytest.raises(ValueError):
        StageEntryTable(1, 'A', ('A', 'B'), (EntryPoint('x', 'C'),))


def test_duplicate_entry_names_are_rejected():
    with pytest.raises(ValueError):
        StageEntryTable(1, 'A', ('A', 'B'),
                        (EntryPoint('x', 'A'), EntryPoint('X', 'B')))


def test_entry_name_may_not_shadow_a_default_alias():
    with pytest.raises(ValueError):
        EntryPoint('auto', 'A')


def test_empty_state_list_is_rejected():
    with pytest.raises(ValueError):
        StageEntryTable(1, 'A', ())


# ----------------------------------------------------------------------
# 日志文本
# ----------------------------------------------------------------------
def test_summary_lists_entry_names():
    summary = make_table().summary()
    for name in ('start', 'step_up', 'ramp', 'corner_2'):
        assert name in summary


def test_describe_lists_states_and_requirements():
    lines = make_table().describe()
    text = '\n'.join(lines)
    assert 'ramp' in text
    assert 'P5_UP_SLOPE' in text
    assert 'requires: 已经站在坡道上' in text
    # 原始状态名也必须可见，否则用状态名调试的人无从知道有哪些
    assert 'P5_TURN_1' in text


def test_entry_for_state():
    table = make_table()
    assert table.entry_for_state('P5_UP_SLOPE').name == 'ramp'
    assert table.entry_for_state('NOPE') is None
