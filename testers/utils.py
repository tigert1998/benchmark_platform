import os
import subprocess


def adb_push(adb_device_id, host_path, guest_path):
    os.system("adb -s {} push {} {}".format(adb_device_id, host_path, guest_path))


def adb_shell(adb_device_id, shell):
    p = subprocess.Popen("adb -s {} shell".format(adb_device_id),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')
