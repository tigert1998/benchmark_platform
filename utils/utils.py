import os
import subprocess
import types
import inspect
import warnings


def escape_path(path):
    for quotation in ["'", '"']:
        if path.startswith(quotation):
            assert path.endswith(quotation)
            return path
    if ' ' in path:
        return '"{}"'.format(path)
    else:
        return path


def shell_with_script(shell, script):
    return "source {} && {}".format(script, shell)


def camel_case_to_snake_case(s):
    ans = []
    last = -1
    for i in range(1, len(s)):
        if 'A' <= s[i] and s[i] <= 'Z':
            ans.append(s[last + 1:i])
            last = i - 1
    ans.append(s[last + 1:])
    return '_'.join(map(lambda s: s.lower(), ans))


def regularize_for_json(obj):
    if isinstance(obj, list):
        new_obj = []
        for i in range(len(obj)):
            new_obj.append(regularize_for_json(obj[i]))
        return new_obj
    elif isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = regularize_for_json(value)
        return new_obj
    elif isinstance(obj, types.FunctionType):
        return inspect.getsource(obj).strip()
    else:
        return obj


def concatenate_flags(flags):

    def to_str(x):
        if isinstance(x, bool):
            return str(x).lower()
        else:
            return str(x)

    res = ''
    for key in flags:
        res += ('--' + key + '=' + to_str(flags[key]) + ' ')
    return res.strip()


def rm_ext(path):
    return os.path.splitext(path)[0]


def set_multilevel_dict(dic, keys, value):
    node = dic
    for key in keys[:-1]:
        if node.get(key) is None:
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value
