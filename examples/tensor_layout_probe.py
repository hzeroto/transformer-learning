"""教师布局演示：追踪元素、stride、复制和逆变换；不实现多头 Attention。"""

import argparse

import torch


def describe(name, x):
    print(name, 'shape:', tuple(x.shape), 'stride:', x.stride(),
          'offset:', x.storage_offset(), 'contiguous:', x.is_contiguous())


def storage_copy_demo():
    """同一块 8 元素存储，观察换轴、复制与非连续 view 的区别。"""
    base = torch.arange(8, dtype=torch.float32)
    x = base.reshape(2, 4)
    y = x.transpose(0, 1)

    def shares_base(t):
        return t.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()

    for name, value in [('x', x), ('y = x.transpose(0,1)', y)]:
        describe(name, value)
        print('  values:', value.tolist(), 'shares base:', shares_base(value))
    assert x.stride() == (4, 1) and y.stride() == (1, 4)
    assert shares_base(x) and shares_base(y)
    addresses = [y.storage_offset() + i * y.stride(0) + j * y.stride(1)
                 for i in range(4) for j in range(2)]
    assert addresses == [0, 4, 1, 5, 2, 6, 3, 7]
    print('y 按行遍历时的底层元素下标:', addresses)

    try:
        y.view(8)
    except RuntimeError:
        print('y.view(8): RuntimeError，地址序列不能由一维固定 stride 表达')
    else:
        raise AssertionError('本例转置后的两轴不能无复制合并')

    flat = y.reshape(8)
    packed = y.contiguous()
    assert flat.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert not shares_base(flat) and not shares_base(packed)
    assert packed.stride() == (2, 1) and torch.equal(packed, y)
    assert x.contiguous() is x
    assert torch.equal(packed.view(8), flat)
    describe('flat = y.reshape(8)', flat)
    describe('packed = y.contiguous()', packed)
    print('flat/packed 各自复制；x.contiguous() 返回 x 自身')

    # 不连续，但地址是等间距的 0、2、4、6，仍能无复制展平。
    spaced = x[:, ::2]
    spaced_flat = spaced.view(4)
    assert not spaced.is_contiguous() and spaced.stride() == (4, 2)
    assert spaced_flat.stride() == (2,) and shares_base(spaced_flat)
    assert spaced_flat.tolist() == [0, 2, 4, 6]
    describe('spaced = x[:, ::2]', spaced)
    describe('spaced.view(4)', spaced_flat)
    print('  values:', spaced_flat.tolist(), 'shares base:', shares_base(spaced_flat))

    tail = x[1:, 1:]
    assert tail.storage_offset() == 5 and shares_base(tail)
    assert tail.is_contiguous() and tail.untyped_storage().nbytes() == 32
    assert tail.numel() * tail.element_size() == 12
    describe('tail = x[1:, 1:]', tail)
    print('tail 逻辑数据 12 字节；仍依赖 base 的整块 32 字节 storage')

    # 修改视图影响共享数据；之前的两份副本不随之变化，无写时复制。
    y[0, 1] = -40
    assert x[1, 0].item() == -40 and base[4].item() == -40
    assert shares_base(y) and flat[1].item() == 4 and packed[0, 1].item() == 4
    print('y[0,1]=-40 后，x[1,0] 同变；flat/packed 中对应值仍为 4')


def main():
    b, t, c, h, dh = 1, 3, 4, 2, 2
    q = torch.arange(12).reshape(b, t, c)
    by_token = q.reshape(b, t, h, dh)
    heads = by_token.transpose(1, 2)
    for name, x in [('Q', q), ('by_token', by_token), ('heads', heads)]:
        describe(name, x)
    print('head 0:', heads[0, 0].tolist())
    print('head 1:', heads[0, 1].tolist())

    try:
        heads.view(b, h * t, dh)
    except RuntimeError:
        print('heads.view(1,6,2): incompatible strides, RuntimeError')
    else:
        raise AssertionError('此指定例子的 H/T 不应可无复制合并')
    print('heads.reshape(1,6,2):', heads.reshape(b, h * t, dh).tolist())
    assert torch.equal(heads.view(b, h, t, dh), heads)
    print('non-contiguous heads.view(same shape): succeeds')

    direct_back = heads.transpose(1, 2)
    describe('inverse transpose of original heads', direct_back)
    assert torch.equal(direct_back.view(b, t, c), q)

    # 只模拟按 (B,H,T,Dh) 连续摆放的新结果，不是一次 Attention 计算。
    y = heads.contiguous()
    describe('simulated contiguous head output', y)
    y_by_token = y.transpose(1, 2)
    describe('head output after transpose', y_by_token)
    try:
        y_by_token.view(b, t, c)
    except RuntimeError:
        print('transposed head output.view(1,3,4): incompatible strides, RuntimeError')
    else:
        raise AssertionError('此指定例子的 H/Dh 不应可无复制合并')
    merged = y_by_token.reshape(b, t, c)
    explicit = y_by_token.contiguous().view(b, t, c)
    assert torch.equal(merged, q) and torch.equal(explicit, q)
    print('correct merge:', merged.tolist())

    bad_heads = q.reshape(b, h, t, dh)
    round_trip = bad_heads.reshape(b, t, c)
    print('direct reshape has requested shape:', tuple(bad_heads.shape))
    print('direct reshape round trip equals Q:', torch.equal(round_trip, q))
    # 不替学习者填写“额外的精确对应关系断言”。

    # 演示共享存储：改动视图会影响 Q，而已复制的 y 保持原值。
    heads[0, 0, 1, 0] = -7
    print('after changing heads[0,0,1,0]: Q[0,1,0] =', q[0, 1, 0].item(),
          '; copied y[0,0,1,0] =', y[0, 0, 1, 0].item())
    assert q[0, 1, 0].item() == -7 and y[0, 0, 1, 0].item() == 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-only', action='store_true', help='只演示存储与复制条件，不重复拆合头')
    args = parser.parse_args()
    if args.storage_only:
        storage_copy_demo()
    else:
        main()
