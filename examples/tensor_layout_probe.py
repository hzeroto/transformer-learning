"""教师布局演示：追踪元素、stride、复制和逆变换；不实现多头 Attention。"""

import torch


def describe(name, x):
    print(name, 'shape:', tuple(x.shape), 'stride:', x.stride(),
          'contiguous:', x.is_contiguous())


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
    main()
