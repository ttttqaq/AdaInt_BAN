"""Face splits with valid-only regional metric aggregation."""
from collections import defaultdict
from mmedit.datasets.registry import DATASETS
from .dataset import Face


@DATASETS.register_module()
class SemanticFace(Face):
    def evaluate(self, results, logger=None):
        if len(results) != len(self):
            raise ValueError('evaluation needs one result per image')
        values = defaultdict(list)
        counts = defaultdict(int)
        for result in results:
            metrics = result['eval_result']
            for name, value in metrics.items():
                if name.startswith('_valid_'):
                    counts[name[7:]] += int(value > 0)
                    continue
                region = ('background' if name.endswith('_background') else
                          'face' if name.endswith('_face') else
                          'foreground' if name.endswith('_foreground') else None)
                if region and metrics.get('_valid_' + region, 1) <= 0:
                    continue
                values[name].append(value)
        output = {name: sum(items) / len(items) for name, items in values.items()}
        for region, count in counts.items():
            output[region + '_valid_count'] = count
        # An entirely absent region has no measurement, not zero error.
        for result in results:
            for name in result['eval_result']:
                if not name.startswith('_valid_') and name not in output:
                    output[name] = float('nan')
        return output
