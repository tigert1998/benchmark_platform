import os
import subprocess


def adb_push(adb_device_id, host_path, guest_path):
    os.system("adb -s {} push {} {}".format(adb_device_id, host_path, guest_path))


def adb_shell(adb_device_id, shell):
    p = subprocess.Popen("adb -s {} shell".format(adb_device_id),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')


def camel_case_to_snake_case(s):
    ans = []
    last = -1
    for i in range(1, len(s)):
        if 'A' <= s[i] and s[i] <= 'Z':
            ans.append(s[last + 1: i])
            last = i - 1
    ans.append(s[last + 1:])
    return '_'.join(map(lambda s: s.lower(), ans))
