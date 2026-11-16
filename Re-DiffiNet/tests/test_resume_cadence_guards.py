"""Standard-library CPU regression checks; never imports the training runtime."""
import ast
import copy
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_guard():
    tree = ast.parse((ROOT / 'testing_utils.py').read_text(encoding='utf-8-sig'))
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == 'validate_checkpoint'
             or isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in ('FORMAT_VERSION', 'METRIC_VERSION')
                 for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'testing_utils.py'), 'exec'), namespace)
    return namespace


class ResumeCadenceGuards(unittest.TestCase):
    def setUp(self):
        self.ns = load_guard()
        self.config = dict(val_every=2, patience=6, model=dict(width=32, sampling_steps=20))
        self.identity = dict(fold=0, training_cases=1000, validation_cases=251)
        self.checkpoint = dict(format_version=self.ns['FORMAT_VERSION'],
                               metric_version=self.ns['METRIC_VERSION'], fold=0,
                               identity=self.identity, config=copy.deepcopy(self.config))

    def validate(self, config):
        self.ns['validate_checkpoint'](self.checkpoint, 0, self.identity, config)

    def test_unchanged_configuration_accepted_without_mutation(self):
        before = copy.deepcopy(self.checkpoint)
        self.validate(self.config)
        self.assertEqual(self.checkpoint, before)

    def test_interval_only_change_rejected_without_mutation(self):
        before = copy.deepcopy(self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'Resume configuration changed'):
            self.validate(dict(self.config, val_every=5))
        self.assertEqual(self.checkpoint, before)

    def test_other_configuration_changes_rejected(self):
        for update in (dict(patience=7), dict(model=dict(width=64, sampling_steps=20)),
                       dict(model=dict(width=32, sampling_steps=4))):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.validate(dict(self.config, **update))

    def test_identity_and_version_guards_remain_strict(self):
        for key, value in [('fold', 1), ('identity', {}), ('format_version', -1),
                           ('metric_version', 'different')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.ns['validate_checkpoint'](dict(self.checkpoint, **{key: value}),
                                                0, self.identity, self.config)


if __name__ == '__main__':
    unittest.main()
