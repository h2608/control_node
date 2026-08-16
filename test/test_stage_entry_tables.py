#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六个赛段真实入口表的一致性测试。

导入赛段模块需要 ROS（rclpy / cv_bridge / tf2_ros），所以这些用例只在
Galactic 容器里跑；主机上 importorskip 会跳过，不会变成红叉。

StageEntryTable 在构造时就校验起点、状态归属和重名，因此“把六张表都造出来”
本身就是主要断言；其余用例锁住跨赛段的公共约定。
"""

import pytest

rclpy = pytest.importorskip('rclpy', reason='stage nodes need a ROS 2 install')

from control_node.stage1_node import p1_entry_table  # noqa: E402
from control_node.stage2_node import (  # noqa: E402
    P2_BALL_SUBCHAIN_STATES,
    P2_CRUISE_STATES,
    p2_entry_table,
)
from control_node.stage3_node import p3_entry_table  # noqa: E402
from control_node.stage4_node import Stage4Node  # noqa: E402
from control_node.stage5_node import Stage5Node  # noqa: E402
from control_node.stage6_node import p6_entry_table  # noqa: E402


def all_tables():
    return [
        p1_entry_table(),
        p2_entry_table(),
        p3_entry_table(),
        Stage4Node.p4_entry_table(),
        Stage5Node.p5_entry_table(),
        p6_entry_table(),
    ]


def test_every_stage_declares_a_table():
    tables = all_tables()
    assert [table.stage_id for table in tables] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize('table', all_tables(), ids=lambda t: 'stage%d' % t.stage_id)
def test_every_table_has_a_start_entry(table):
    """每个赛段都必须有一个叫 start 的入口，且它就是正常起点。"""
    entry = dict((e.name, e) for e in table.entry_points).get('start')
    assert entry is not None
    assert entry.state == table.start_state


@pytest.mark.parametrize('table', all_tables(), ids=lambda t: 'stage%d' % t.stage_id)
def test_default_request_matches_the_previous_hardcoded_start(table):
    assert table.resolve('default').state == table.start_state
    assert table.resolve('start').state == table.start_state


@pytest.mark.parametrize('table', all_tables(), ids=lambda t: 'stage%d' % t.stage_id)
def test_every_entry_point_resolves_to_itself(table):
    for entry in table.entry_points:
        resolution = table.resolve(entry.name)
        assert resolution.ok is True
        assert resolution.state == entry.state


@pytest.mark.parametrize('table', all_tables(), ids=lambda t: 'stage%d' % t.stage_id)
def test_every_state_is_reachable_by_its_own_name(table):
    for state in table.state_names():
        assert table.resolve(state).state == state


def test_stage1_states_match_the_state_machine():
    assert p1_entry_table().resolve('cruise').state == 'P1_STAGE1_CRUISE'
    assert p1_entry_table().resolve('ball').state == 'P1_APPROACH_BLUE_BALL'


def test_stage5_ramp_is_the_up_slope_section():
    """用户举例的“只跑第五赛段的坡道段”。"""
    table = Stage5Node.p5_entry_table()
    assert table.resolve('ramp').state == Stage5Node.P5_UP_SLOPE
    # 坡道段前面的所有状态都被跳过了，所以必须有明确的摆位前提。
    assert table.resolve('ramp').requires


def test_stage5_entry_names_cover_the_route_model_segments():
    """入口名和 route_model 的段名保持同一套词，证据日志才对得上。"""
    from control_node.route_model import RouteModel

    model = RouteModel()
    table = Stage5Node.p5_entry_table()
    for name, expected_segment in (
            ('step_up', 'entry_step_up'),
            ('ramp', 'up_slope'),
            ('corner_1', 'corner_1'),
            ('straight_1', 'straight_1'),
            ('corner_2', 'corner_2'),
            ('straight_2', 'straight_2'),
            ('corner_3', 'corner_3'),
            ('straight_3', 'straight_3'),
            ('corner_4', 'corner_4'),
            ('descent', 'right_descent'),
            ('final', 'final_zone'),
    ):
        state = table.resolve(name).state
        segment = model.segment_for_state(state)
        assert segment is not None, name
        assert segment.name == expected_segment, name


def test_stage2_ball_subchain_entries_are_flagged():
    """撞球子链入口必须提示 p2_ball_return_state，否则子链会回到自己。"""
    table = p2_entry_table()
    for name in ('ball_align', 'ball_hit', 'ball_shift'):
        resolution = table.resolve(name)
        assert resolution.state in P2_BALL_SUBCHAIN_STATES
        assert any('p2_ball_return_state' in item for item in resolution.requires)


def test_stage2_cruise_states_are_entry_points():
    table = p2_entry_table()
    reachable = set(entry.state for entry in table.entry_points)
    for state in P2_CRUISE_STATES:
        assert state in reachable


def test_stage4_dashed_side_entries_are_flagged():
    table = Stage4Node.p4_entry_table()
    for name in ('obstacle_route', 'post_hit_obstacle'):
        resolution = table.resolve(name)
        assert any('dashed_side' in item for item in resolution.requires)


def test_stage4_table_states_are_the_legacy_valid_states():
    """入口表不能悄悄放宽原有的合法 initial_state 集合。"""
    table = Stage4Node.p4_entry_table()
    assert list(table.state_names()) == Stage4Node.get_all_state_names()


def test_stage6_test_mode_moves_the_default_start():
    assert p6_entry_table().start_state == 'BLIND_MARCH'
    assert p6_entry_table('NORTHWARD_MARCH').start_state == 'NORTHWARD_MARCH'
    # 具名入口不随之改变：start 始终是完整流程的起点。
    assert p6_entry_table('NORTHWARD_MARCH').resolve('start').state == 'BLIND_MARCH'
