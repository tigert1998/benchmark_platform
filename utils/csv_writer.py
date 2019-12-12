class CSVWriter:
    def __init__(self):
        self.fd = None
        self.previous_filename = None

    def _write_titles(self, titles):
        self.fd.write(','.join(titles) + '\n')
        self.fd.flush()

    def _write_data(self, data):
        self.fd.write(','.join(map(str, data)) + '\n')
        self.fd.flush()

    def _close(self):
        if self.fd is not None:
            self.fd.close()

    def update_data(self, filename, titles, data, is_resume):
        if self.previous_filename is None:
            if is_resume:
                self.fd = open(filename, 'a')
            else:
                self.fd = open(filename, 'w')
                self._write_titles(titles)
            self._write_data(data)
        elif self.previous_filename != filename:
            self._close()
            self.fd = open(filename, 'w')
            self._write_titles(titles)
            self._write_data(data)
        else:
            self._write_data(data)
        self.previous_filename = filename

    def __del__(self):
        self._close()
