"""在一维存储位置和高阶张量索引之间转换。"""


def offset(shape: tuple[int, ...], indices: tuple[int, ...]) -> int:
    """把多轴索引转换为一维存储位置。

    Args:
        shape: 每个轴的长度；必须非空，且所有长度均为正整数。
        indices: 每个轴上的索引，数量必须与 shape 中的轴数一致。

    Raises:
        ValueError: shape 非法，或 indices 的数量与轴数不同。
        IndexError: 某个索引超出对应轴的范围。
    """
    if len (shape) == 0 or any(s <= 0 for s in shape):
        raise ValueError("shape 非法")
    if len(indices) != len(shape):
        raise ValueError("indices 的数量与轴数不同")
    ans = 0
    pre = 1
    sizeAfter = []
    for s in reversed(shape):
        sizeAfter.append(pre)
        pre *= s
    lenth = len(shape)
    sizeAfter.reverse()
    for i in range(lenth):
        if indices[i] < 0 or indices[i] >= shape[i]:
            raise IndexError("某个索引超出对应轴的范围")
        ans += indices[i] * sizeAfter[i]
    return ans


def indices(shape: tuple[int, ...], offset: int) -> tuple[int, ...]:
    """把一维存储位置转换为多轴索引。

    Args:
        shape: 每个轴的长度；必须非空，且所有长度均为正整数。
        offset: 一维存储位置。

    Raises:
        ValueError: shape 非法。
        IndexError: offset 超出张量元素总数的范围。
    """
    if shape == () or any(s <= 0 for s in shape):
        raise ValueError("shape 非法")
    total_elements = 1
    sizeAfter = []
    for s in reversed(shape):
        sizeAfter.append(total_elements)
        total_elements *= s
    if offset < 0 or offset >= total_elements:
        raise IndexError("offset 超出张量元素总数的范围")

    ans = []
    for s in reversed(sizeAfter):
        ans.append(offset // s)
        offset -= (offset // s) * s
    return tuple(ans)
