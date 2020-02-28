class CSVWriter:
    """A non-standard CSV Writer. 
    It will generate titles in the normal columns if the current title
    does not match with the former one.
    """

    def __init__(self):
        self.fd = None
        self.previous_filename = None
        self.previous_titles = None

    def _write_titles(self, titles):
        self.fd.write(','.join(titles) + '\n')
        self.fd.flush()

    def _write_data(self, data):
        self.fd.write(','.join(map(str, data)) + '\n')
        self.fd.flush()

    def _close(self):
        if self.fd is not None:
            self.fd.close()

    def update_data(self, filename, data, is_resume):
        if self.previous_filename is None:
            if is_resume:
                self.fd = open(filename, 'a')
            else:
                self.fd = open(filename, 'w')
            self._write_titles(data.keys())
            self._write_data(data.values())
        elif self.previous_filename != filename:
            self._close()
            self.fd = open(filename, 'w')
            self._write_titles(data.keys())
            self._write_data(data.values())
        elif self.previous_titles != data.keys():
            self._write_titles(data.keys())
            self._write_data(data.values())
        else:
            self._write_data(data.values())
        self.previous_filename = filename
        self.previous_titles = data.keys()

    def __del__(self):
        self._close()
