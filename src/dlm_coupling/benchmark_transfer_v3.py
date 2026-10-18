"""Concrete output labels; data, reasoning allowance and v2 scoring stay fixed."""
from . import benchmark_transfer_v2 as previous

PROTOCOL = 'benchmark-transfer-v3-concrete-labels'


def prompt(case, requested, context='full'):
    text = previous.prompt(case, requested, context)
    old = 'as LABEL: short answer.'
    labels = ', '.join(f'{label}:' for label in requested)
    text = text.replace(old, f'using the labels {labels}, respectively.')
    template = '\n'.join(f'{label}: [answer to question {label}]' for label in requested)
    return text + ('\n\nFinal answer template (replace each bracketed placeholder with your short answer):\n'
                   'FINAL:\n' + template)
