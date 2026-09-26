"""Harness-owned integer RN-even binary32 oracle; no candidate imports."""


def classify(word):
    magnitude = word & 0x7fffffff
    return 'nan' if magnitude > 0x7f800000 else 'inf' if magnitude == 0x7f800000 else 'finite'


def add32(left, right):
    """One IEEE binary32 addition, integers only, canonical diagnostic NaN."""
    lk, rk = classify(left), classify(right)
    if lk == 'nan' or rk == 'nan':
        return 0x7fc00000
    if lk == 'inf' or rk == 'inf':
        if lk == rk and ((left ^ right) & 0x80000000):
            return 0x7fc00000
        return left if lk == 'inf' else right
    def exact(word):
        exponent = (word >> 23) & 255
        significand = word & 0x7fffff
        if exponent:
            significand |= 0x800000
        integer = significand << max(exponent - 1, 0)
        return -integer if word & 0x80000000 else integer
    total = exact(left) + exact(right)
    if total == 0:
        return 0x80000000 if left == right == 0x80000000 else 0
    sign = 0x80000000 if total < 0 else 0
    magnitude = abs(total)
    discarded = max(magnitude.bit_length() - 24, 0)
    rounded = magnitude >> discarded
    if discarded:
        remainder = magnitude - (rounded << discarded)
        halfway = 1 << (discarded - 1)
        if remainder > halfway or (remainder == halfway and (rounded & 1)):
            rounded += 1
    if rounded >= 0x1000000:
        rounded >>= 1
        discarded += 1
    if rounded < 0x800000:
        return sign | rounded
    exponent = discarded + 1
    if exponent >= 255:
        return sign | 0x7f800000
    return sign | (exponent << 23) | (rounded & 0x7fffff)


def recurrence(bf16_timeline):
    result = 0
    for value in bf16_timeline:
        if type(value) is not int or not 0 <= value <= 65535:
            raise ValueError('BF16_BITS_REQUIRED')
        result = add32(result, value << 16)
    return result
