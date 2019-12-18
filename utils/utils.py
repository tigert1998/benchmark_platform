import os
import subprocess
import types
import inspect


def escape_path(path):
    if path.startswith('"'):
        assert path.endswith('"')
        return path
    if path.startswith("'"):
        assert path.endswith("'")
        return path
    if ' ' in path:
        return '"{}"'.format(path)


def adb_push(adb_device_id, host_path, guest_path):
    os.system("adb -s {} push {} {}".format(
        adb_device_id,
        escape_path(host_path),
        guest_path
    ))


def adb_pull(adb_device_id, guest_path, host_path):
    os.system("adb -s {} pull {} {}".format(adb_device_id, guest_path, host_path))


def adb_shell(adb_device_id, shell, su=False):
    p = subprocess.Popen("adb -s {} shell {}".format(adb_device_id, "su" if su else ""),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')


def adb_shell_su(adb_device_id, shell):
    return adb_shell(adb_device_id, shell, True)


def shell_with_script(shell, script):
    return "source {} && {}".format(script, shell)


def inquire_adb_device(adb_device_id):
    getprop_items = [
        "ro.product.model",
        "ro.build.version.release",
        "ro.build.version.sdk",
    ]
    res = {"adb_device_id": adb_device_id}
    for item in getprop_items:
        res[item] = adb_shell(adb_device_id, "getprop {}".format(item)).strip()
    return res


def camel_case_to_snake_case(s):
    ans = []
    last = -1
    for i in range(1, len(s)):
        if 'A' <= s[i] and s[i] <= 'Z':
            ans.append(s[last + 1: i])
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
    return ".".join(path.split(".")[:-1])
