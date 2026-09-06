"""The deployment-wide `autostop` default in the SkyPilot config.

A top-level `autostop:` bounds every cluster launched without one of its own —
the lever an operator has for idle clusters on a backend where the alternative
is waiting out a batch scheduler's walltime.
"""

import pytest

from sky import execution
from sky import skypilot_config
from sky.skylet import autostop_lib
from sky.utils import common_utils
from sky.utils import schemas


class TestDefaultAutostopConfig:

    def test_unset_means_no_autostop(self):
        # pylint: disable=protected-access
        assert execution._default_autostop_config() is None

    def test_mapping_form(self):
        with skypilot_config.override_skypilot_config(
            {'autostop': {
                'idle_minutes': 30,
                'down': True
            }}):
            # pylint: disable=protected-access
            config = execution._default_autostop_config()
        assert config is not None
        assert config.enabled
        assert config.idle_minutes == 30
        assert config.down

    def test_shorthands(self):
        with skypilot_config.override_skypilot_config({'autostop': 15}):
            # pylint: disable=protected-access
            config = execution._default_autostop_config()
        assert config is not None and config.idle_minutes == 15
        # Down defaults to false: the shorthand asks to stop, and a cloud that
        # cannot stop should say so rather than silently terminate.
        assert not config.down

        with skypilot_config.override_skypilot_config({'autostop': False}):
            # pylint: disable=protected-access
            config = execution._default_autostop_config()
        assert config is not None and not config.enabled

    def test_wait_for(self):
        with skypilot_config.override_skypilot_config(
            {'autostop': {
                'idle_minutes': 5,
                'wait_for': 'jobs_and_ssh',
            }}):
            # pylint: disable=protected-access
            config = execution._default_autostop_config()
        assert config is not None
        assert config.wait_for == autostop_lib.AutostopWaitFor.JOBS_AND_SSH

    @pytest.mark.parametrize('value', [
        30,
        '30m',
        False,
        {
            'idle_minutes': 30,
            'down': True
        },
    ])
    def test_schema_accepts_the_same_shapes_as_resources(self, value):
        common_utils.validate_schema({'autostop': value},
                                     schemas.get_config_schema(), '')

    def test_schema_rejects_unknown_fields(self):
        with pytest.raises(Exception):
            common_utils.validate_schema({'autostop': {
                'idle_minute': 30
            }}, schemas.get_config_schema(), '')
