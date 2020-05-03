import subprocess
import os
import platform

from .utils import escape_path
from .class_with_settings import ClassWithSettings


class Connection(ClassWithSettings):
    def push(self, local_path: str, remote_path: str):
        pass

    def pull(self, remote_path: str, local_path: str):
        pass

    def shell(self, shell: str):
        if "Win" in platform.platform():
            shell_exe = "powershell"
        else:
            shell_exe = os.environ["SHELL"]
        p = subprocess.Popen(
            shell_exe,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')

    def brief(self):
        return "local"

    def snapshot(self):
        return {
            "remark": "local"
        }


class Ssh(Connection):
    def __init__(self, address):
        self.address = address

    def push(self, local_path: str, remote_path: str):
        assert 0 == os.system("scp {} {}:{}".format(local_path, self.address,
                                                    remote_path))

    def pull(self, remote_path: str, local_path: str):
        assert 0 == os.system("scp {}:{} {}".format(self.address, remote_path,
                                                    local_path))

    def shell(self, shell: str):
        p = subprocess.Popen(
            ["ssh", self.address, "bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')

    def snapshot(self):
        return {
            "address": self.address,
            "uname -a": self.shell("uname -a").strip()
        }

    def brief(self):
        return self.address


class Adb(Connection):
    def __init__(self, adb_device_id: str, su: bool):
        self.adb_device_id = adb_device_id
        self.su = su

    def push(self, local_path: str, remote_path: str):
        assert 0 == os.system("adb -s {} push {} {}".format(
            self.adb_device_id, escape_path(local_path), remote_path))

    def pull(self, remote_path: str, local_path: str):
        assert 0 == os.system("adb -s {} pull {} {}".format(
            self.adb_device_id, remote_path, local_path))

    def shell(self, shell: str):
        p = subprocess.Popen(
            ["adb", "-s", self.adb_device_id, "shell", "su" if self.su else ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        return p.communicate(bytes(shell, 'utf-8'))[0].decode('utf-8')

    def query_battery(self):
        s = self.shell("dumpsys battery")
        ans = s.split('\n')
        ans = map(lambda s: list(map(lambda s: s.strip(), s.split(':'))), ans)
        ans = filter(lambda arr: len(arr) >= 2 and len(arr[1]) >= 1, ans)
        return {key: value for key, value in ans}

    def snapshot(self):
        getprop_items = [
            "ro.product.model",
            "ro.build.version.release",
            "ro.build.version.sdk",
        ]
        res = {"adb_device_id": self.adb_device_id}
        for item in getprop_items:
            res[item] = self.shell("getprop {}".format(item)).strip()
        return res

    def brief(self):
        return self.adb_device_id
