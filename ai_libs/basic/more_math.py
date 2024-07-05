import hashlib


def to_base_n(x: int, base: int, output_length=None) -> list:
    assert x >= 0, f"Invalid input x {x}"
    assert base >= 2, f"Invalid input base {base}"
    res = []
    while True:
        cur = x % base
        res.append(cur)
        x //= base
        if (output_length is None and x == 0) or (len(res) == output_length):
            break
    return res[::-1]


def to_base32_str(x: int, output_length=None) -> str:
    assert x >= 0, f"Invalid input {x}"
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    nums = to_base_n(x, 32, output_length)
    res = [digits[i] for i in nums]
    res = ''.join(res)
    return res


def to_base60_str(x: int, output_length=None) -> str:
    assert x >= 0, f"Invalid input {x}"
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    nums = to_base_n(x, 60, output_length)
    res = [digits[i] for i in nums]
    res = ''.join(res)
    return res


def test_to_base32():
    a = to_base32_str(9)
    assert a == '9'
    a = to_base32_str(32)
    assert a == '10'
    a = to_base32_str(34, output_length=3)
    assert a == '012'


def test_to_base60():
    a = to_base60_str(9)
    assert a == '9'
    a = to_base60_str(60)
    assert a == '10'
    a = to_base60_str(64, output_length=3)
    assert a == '014'


if __name__ == "__main__":
    print(to_base60_str(0))
