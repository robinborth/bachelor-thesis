from src.validation.build_cira_dataset import labels2cirapairs


def test_labels2cirapairs():
    labels = '[{"name": "Cause1", "begin": 0, "end": 29}, {"name": "Effect1", "begin": 37, "end": 46}]'
    pairs = labels2cirapairs(labels)
    assert len(pairs) == 1
    ((cause1, effect1),) = pairs
    assert len(cause1) == len(effect1) == 1

    labels = '[{"name": "Cause1", "begin": 0, "end": 50}, {"name": "Cause1", "begin": 139, "end": 147}, {"name": "Cause1", "begin": 151, "end": 173}, {"name": "Cause1", "begin": 176, "end": 179}, {"name": "Effect1", "begin": 60, "end": 63}, {"name": "Effect1", "begin": 95, "end": 110}, {"name": "Effect2", "begin": 60, "end": 63}, {"name": "Effect2", "begin": 64, "end": 94}, {"name": "Effect2", "begin": 95, "end": 110}, {"name": "Effect2", "begin": 115, "end": 124}, {"name": "Effect3", "begin": 180, "end": 216}]'
    pairs = labels2cirapairs(labels)
    assert len(pairs) == 1
    ((cause1, effect1),) = pairs
    assert len(cause1) == 4
    assert len(effect1) == 2
