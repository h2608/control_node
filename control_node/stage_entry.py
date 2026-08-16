#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛段调试入口（entry point）解析。

纯 Python：不导入 ROS、OpenCV 或 numpy，因此可以在没有 Galactic 的主机上
直接用 pytest 跑（见 ``test/test_stage_entry.py``），和 ``route_model.py`` /
``deck_lateral.py`` 的做法一致。

每个赛段节点声明一张 :class:`StageEntryTable`：把内部状态机里可以安全直接
进入的状态，映射成一小组人类可读的调试入口名（例如第五赛段的 ``ramp`` 就是
上坡段 ``P5_UP_SLOPE``）。这样调试某一段流程时不必记住内部状态名：

    ros2 launch control_node full_competition.launch.py \\
        single_stage:=true start_stage:=5 stage5_entry:=ramp

原有的 ``<赛段>_initial_state`` 参数继续可用（直接写状态名），本模块只是
在它上面加了统一的命名、校验、回退与日志文本。

注意：入口名只决定状态机从哪个状态开始。**机器人本体必须已经被摆到该状态
对应的位置与朝向**——赛段代码不会把狗搬过去。
"""

# ``entry_point`` 参数取这些值时表示“不指定”，使用赛段自己的正常起点。
DEFAULT_ALIASES = ('', 'default', 'none', 'auto')

# EntryResolution.source 的取值。
SOURCE_DEFAULT = 'default'
SOURCE_ENTRY_POINT = 'entry_point'
SOURCE_STATE = 'state'
SOURCE_FALLBACK = 'fallback'


def is_default_request(requested):
    """True 表示调用方没有真正指定入口（空 / default / none / auto）。"""
    if requested is None:
        return True
    return _normalize(requested) in DEFAULT_ALIASES


def _normalize(name):
    """入口名归一化：去空白、小写、连字符当下划线。"""
    return str(name).strip().lower().replace('-', '_')


class EntryPoint(object):
    """一个具名调试入口。

    name        小写入口名，命令行里写的就是它。
    state       进入的状态机状态名。
    description 一句话说明这个入口从赛段的哪一段开始。
    requires    进入前必须额外满足的前提（会在日志里逐条打印）。
    """

    __slots__ = ('name', 'state', 'description', 'requires')

    def __init__(self, name, state, description='', requires=()):
        self.name = _normalize(name)
        if not self.name:
            raise ValueError('entry point name must not be empty')
        if self.name in DEFAULT_ALIASES:
            raise ValueError(
                "entry point name '{}' collides with a default alias".format(
                    self.name))
        self.state = str(state)
        self.description = str(description)
        self.requires = tuple(str(item) for item in requires)

    def __repr__(self):  # pragma: no cover - debug aid only
        return 'EntryPoint({}->{})'.format(self.name, self.state)


class EntryResolution(object):
    """一次入口解析的结果。"""

    __slots__ = ('stage_id', 'requested', 'state', 'entry_point', 'source',
                 'ok', 'message')

    def __init__(self, stage_id, requested, state, entry_point, source, ok,
                 message):
        self.stage_id = int(stage_id)
        self.requested = str(requested)
        self.state = str(state)
        self.entry_point = entry_point
        self.source = str(source)
        self.ok = bool(ok)
        self.message = str(message)

    @property
    def is_default(self):
        """True 表示走的是赛段正常起点，不是调试入口。"""
        return self.source == SOURCE_DEFAULT

    @property
    def requires(self):
        return self.entry_point.requires if self.entry_point is not None else ()

    def to_dict(self):
        return {
            'stage_id': self.stage_id,
            'requested': self.requested,
            'state': self.state,
            'entry_point': (self.entry_point.name
                            if self.entry_point is not None else ''),
            'source': self.source,
            'ok': self.ok,
            'message': self.message,
        }

    def __repr__(self):  # pragma: no cover - debug aid only
        return 'EntryResolution({}, {}, ok={})'.format(
            self.stage_id, self.state, self.ok)


class StageEntryTable(object):
    """一个赛段的调试入口表。

    stage_id     赛段编号（只用于日志）。
    start_state  该赛段的正常起点状态。
    states       允许作为入口的全部状态名，按流程顺序给出。
    entry_points :class:`EntryPoint` 序列，同样按流程顺序。

    构造时就校验：起点和每个入口的状态都必须在 ``states`` 里，入口名不能重复。
    这些是编码错误而不是运行期输入错误，所以直接抛 ``ValueError``，由单元测试
    在没有机器人的情况下抓住。
    """

    def __init__(self, stage_id, start_state, states, entry_points=()):
        self.stage_id = int(stage_id)
        self.start_state = str(start_state)

        ordered_states = []
        for state in states:
            state = str(state)
            if state not in ordered_states:
                ordered_states.append(state)
        self.states = tuple(ordered_states)
        if not self.states:
            raise ValueError(
                'stage {} entry table has no states'.format(self.stage_id))
        if self.start_state not in self.states:
            raise ValueError(
                "stage {} start_state '{}' is not one of its states".format(
                    self.stage_id, self.start_state))

        self.entry_points = tuple(entry_points)
        self._by_name = {}
        for entry in self.entry_points:
            if entry.name in self._by_name:
                raise ValueError(
                    "stage {} declares entry point '{}' twice".format(
                        self.stage_id, entry.name))
            if entry.state not in self.states:
                raise ValueError(
                    "stage {} entry point '{}' targets unknown state "
                    "'{}'".format(self.stage_id, entry.name, entry.state))
            self._by_name[entry.name] = entry
        self._by_state = dict((state.upper(), state) for state in self.states)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def entry_names(self):
        return tuple(entry.name for entry in self.entry_points)

    def state_names(self):
        return self.states

    def entry_for_state(self, state):
        """返回指向该状态的第一个具名入口（没有则 None）。"""
        for entry in self.entry_points:
            if entry.state == state:
                return entry
        return None

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def resolve(self, requested):
        """把一个请求（入口名 / 状态名 / 空）解析成状态名。

        非法请求不抛异常：返回 ``ok=False`` 并回退到赛段正常起点，让调试参数
        写错的运行仍然是一次可预期的正常运行，而不是一个起不来的节点。
        """
        raw = '' if requested is None else str(requested).strip()
        key = _normalize(raw)

        if key in DEFAULT_ALIASES:
            entry = self.entry_for_state(self.start_state)
            return EntryResolution(
                self.stage_id, raw, self.start_state, entry, SOURCE_DEFAULT,
                True,
                'stage {} starts at its normal entry {}'.format(
                    self.stage_id, self.start_state))

        entry = self._by_name.get(key)
        if entry is not None:
            return EntryResolution(
                self.stage_id, raw, entry.state, entry, SOURCE_ENTRY_POINT,
                True,
                "stage {} DEBUG ENTRY '{}' -> {}{}".format(
                    self.stage_id, entry.name, entry.state,
                    ' ({})'.format(entry.description)
                    if entry.description else ''))

        state = self._by_state.get(raw.upper())
        if state is not None:
            return EntryResolution(
                self.stage_id, raw, state, self.entry_for_state(state),
                SOURCE_STATE, True,
                'stage {} DEBUG ENTRY at state {}'.format(
                    self.stage_id, state))

        return EntryResolution(
            self.stage_id, raw, self.start_state, None, SOURCE_FALLBACK, False,
            "stage {} entry '{}' is neither an entry point nor a state; "
            'falling back to {}'.format(self.stage_id, raw, self.start_state))

    # ------------------------------------------------------------------
    # 日志文本
    # ------------------------------------------------------------------
    def summary(self):
        """一行入口名清单，正常启动时打印。"""
        names = self.entry_names()
        if not names:
            return 'stage {} declares no named entry points'.format(
                self.stage_id)
        return 'stage {} entry points: {}'.format(
            self.stage_id, ', '.join(names))

    def describe(self):
        """逐行入口说明，入口写错时打印。"""
        lines = ['stage {} entry points (name -> state):'.format(self.stage_id)]
        width = max([len(name) for name in self.entry_names()] or [1])
        for entry in self.entry_points:
            line = '  {} -> {}'.format(entry.name.ljust(width), entry.state)
            if entry.description:
                line += '  # ' + entry.description
            lines.append(line)
            for requirement in entry.requires:
                lines.append('  {}    requires: {}'.format(
                    ' ' * width, requirement))
        lines.append(
            '  any raw state name is also accepted: {}'.format(
                ', '.join(self.states)))
        return lines
