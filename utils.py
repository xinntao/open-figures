'''utils for plotting'''
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_data_from_tensorboard(log_path, tag):
    '''
    Get raw data (steps and values) from tensorboard events
    '''
    # tensorboard event
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    scalar_list = event_acc.Tags()['scalars']
    print('tag list: ', scalar_list)
    steps = [int(s.step) for s in event_acc.Scalars(tag)]
    values = [s.value for s in event_acc.Scalars(tag)]
    return steps, values


def read_data_from_txt_2p(path, pattern, step_one=False):
    '''two patterns
    step_one: add 1 to steps'''
    with open(path) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    steps = []
    values = []

    pattern = re.compile(pattern)
    for l in lines:
        match = pattern.match(l)
        if match:
            steps.append(int(match.group(1)))
            values.append(float(match.group(2)))
    if step_one:
        steps = [v + 1 for v in steps]
    return steps, values


def read_data_from_txt_1p(path, pattern):
    '''only one pattern'''
    with open(path) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    data = []

    pattern = re.compile(pattern)
    for l in lines:
        match = pattern.match(l)
        if match:
            data.append(float(match.group(1)))
    return data


def smooth_data(values, smooth_weight):
    '''
    Smooth data using 1st-order IIR low-pass filter (what tensorflow does).
    https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/\
        tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts#L704
    '''
    values_sm = []
    last_sm_value = values[0]
    for value in values:
        value_sm = last_sm_value * smooth_weight + (1 - smooth_weight) * value
        values_sm.append(value_sm)
        last_sm_value = value_sm
    return values_sm
