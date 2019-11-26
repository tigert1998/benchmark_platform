def rfind_assign(s, mark):
    mark += "="
    l_idx = s.rfind(mark) + len(mark)
    r_idx = l_idx
    while s[r_idx] not in [' ', '\n']:
        r_idx += 1
    return s[l_idx: r_idx]


def rfind_assign_float(s, mark):
    return float(rfind_assign(s, mark))


def rfind_assign_int(s, mark):
    return int(rfind_assign(s, mark))


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


def table_try_float(table):
    for i in range(len(table)):
        for j in range(len(table[i])):
            try:
                table[i][j] = float(table[i][j])
            except:
                pass
    return table
