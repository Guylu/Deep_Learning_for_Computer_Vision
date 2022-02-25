import unittest

import torch
import torch.nn.functional as F

# import functional
from functional import linear, linear_backward
from functional import relu, relu_backward
from functional import softmax, softmax_backward
from functional import cross_entropy, cross_entropy_backward


class TestLinear(unittest.TestCase):

  def testContext(self):
    ctx = [None]
    x = torch.randn(4, 5)
    w = torch.randn(10, 5)
    b = torch.randn(10)
    y = linear(x, w, b, ctx=ctx)
    # test append to ctx
    assert len(ctx) == 2 and ctx[-1] is not None, 'backward call was not appended to ctx'
    lst_ctx = ctx[-1]
    # test backward_fn, args
    assert len(lst_ctx) == 2, '(backwad_fn, args) should be added ctx'
    backwad_fn, args = lst_ctx
    # test backward_fn
    assert backwad_fn is linear_backward, 'wrong backward_fn was added to ctx'
    # test args
    ref_args = [('y', y), ('x', x), ('w', w), ('b', b)]
    assert len(args) == len(ref_args), 'incorrect number of args in ctx'
    for i, (name, var) in enumerate(ref_args):
      assert args[i] is var, f'args[{i}] should be {name}'

  def testShape(self):
    x = torch.randn(4, 5)
    w = torch.randn(10, 5)
    b = torch.randn(10)
    y = linear(x, w, b)
    expected = (4, 10)
    dbg = f'got: {y.shape}. expected: {expected}.'
    assert y.shape == expected, f'output shape is wrong. {dbg}'

  def testForward(self):
    x = torch.randn(4, 5)
    w = torch.randn(3, 5)
    b = torch.randn(3)
    y = linear(x, w, b)
    ref = F.linear(x, w, b)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output.\n{dbg}')

  def testForwardNoBatch(self):
    x = torch.randn(1, 5)
    w = torch.randn(3, 5)
    b = torch.randn(3)
    y = linear(x, w, b)
    ref = F.linear(x, w, b)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output.\n{dbg}')

  def testBackward(self):
    x = torch.tensor([[ 1.1362295151, -0.1957436055,  1.2812129259],
                      [-0.4800147414, -0.1578529775, -0.1595831513],
                      [-1.1546111107, -0.0174976680, -1.4397902489],
                      [ 0.4418801367, -0.3355576396,  1.2962621450],
                      [ 0.1753998846,  0.1305148751,  0.3164737225]])
    w = torch.tensor([[-0.0345642045,  0.1318718493,  2.0589654446],
                      [ 0.4276058972, -0.0082573155, -0.2064828724],
                      [ 0.2564657331, -0.9770386815,  0.3094902337],
                      [-0.5476234555, -0.3164663911,  0.8526095748]])
    b = torch.tensor([1.4529314041, 0.7067053914, 0.8074156642, 0.1518988013])
    y = torch.tensor([[ 4.0258188248,  0.9296316504,  1.6865916252,  0.6839935184],
                      [ 1.1201301813,  0.5357028842,  0.7891473770,  0.3286591768],
                      [-1.4739460945,  0.5104233623,  0.0827923417, -0.4378505349],
                      [ 4.0623664856,  0.6307708025,  1.6497759819,  1.1213130951],
                      [ 2.1156885624,  0.7152833343,  0.8228271604,  0.2843706608]])
    y.grad = torch.tensor([[ 1.5416468382,  0.4738827050, -0.5827679634,  0.8227812052],
                          [ 0.4922323823, -1.6976156235,  0.1881256998,  0.3348745406],
                          [ 1.4288100004,  1.7683039904,  1.4654935598, -1.2190614939],
                          [ 0.3194803298,  1.1692222357,  0.4189493656, -0.2167775631],
                          [-0.5232126713,  1.5055997372, -0.2402084321,  0.6949081421]])
    ref_x_grad = torch.tensor([[-0.4506850839,  0.5083910227,  3.5974991322],
                              [-0.8780614138, -0.2108532786,  1.7077581882],
                              [ 1.7501870394, -0.8722335100,  1.9909185171],
                              [ 0.7150824070, -0.3082510829,  0.3612086773],
                              [ 0.2197344899, -0.0666513741, -0.8700141907]])
    ref_w_grad = torch.tensor([[-0.0849334896, -0.5799597502,  0.0879863650],
                              [ 0.0923547223, -0.0515653044,  0.3241699934],
                              [-2.3015434742, -0.1131982580, -2.4196264744],
                              [ 2.2077627182, -0.0291471574,  2.6948299408]])
    ref_b_grad = torch.tensor([3.2589571476, 3.2193930149, 1.2495923042, 0.4167248011])
    for t in (x, w, b):
      t.grad = torch.zeros_like(t)
    linear_backward(y, x, w, b)
    to_test = [
      ('x', x, ref_x_grad),
      ('w', w, ref_w_grad),
      ('b', b, ref_b_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')

  def testBackwardNoBatch(self):
    x = torch.tensor([[-0.9918826818,  0.4405492246,  0.2634261250]])
    w = torch.tensor([[ 0.1126472279, -1.3489197493, -0.6054549217],
                      [-1.1746079922, -0.4836844206,  0.2167130411],
                      [-0.1777121872,  1.9724557400,  0.3239948452],
                      [-1.0432158709, -0.1252886653,  1.3490948677]])
    b = torch.tensor([ 0.3346850872, -0.5169801116, -1.2668093443,  0.5537579060])
    y = torch.tensor([[-0.5308059454,  0.4920942187, -0.1362271309,  1.8886966705]])
    y.grad = torch.tensor([[ 0.8985987306,  0.4850614369,  3.1352078915, -2.3355576992]])
    ref_x_grad = torch.tensor([[ 1.4107939005,  5.0299234390, -2.5740394592]])
    ref_w_grad = torch.tensor([[-0.8913044930,  0.3958769739,  0.2367143780],
                               [-0.4811240435,  0.2136934400,  0.1277778596],
                               [-3.1097583771,  1.3812134266,  0.8258956671],
                               [ 2.3165991306, -1.0289281607, -0.6152468920]])
    ref_b_grad = torch.tensor([ 0.8985987306,  0.4850614369,  3.1352078915, -2.3355576992])
    for t in (x, w, b):
      t.grad = torch.zeros_like(t)
    linear_backward(y, x, w, b)
    to_test = [
      ('x', x, ref_x_grad),
      ('w', w, ref_w_grad),
      ('b', b, ref_b_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')


class TestReLU(unittest.TestCase):

  def testContext(self):
    ctx = [None]
    x = torch.randn(4, 5, 6)
    y = relu(x, ctx=ctx)
    # test append to ctx
    assert len(ctx) == 2 and ctx[-1] is not None, 'backward call was not appended to ctx'
    lst_ctx = ctx[-1]
    # test backward_fn, args
    assert len(lst_ctx) == 2, '(backwad_fn, args) should be added ctx'
    backwad_fn, args = lst_ctx
    # test backward_fn
    assert backwad_fn is relu_backward, 'wrong backward_fn was added to ctx'
    # test args
    ref_args = [('y', y), ('x', x)]
    assert len(args) == len(ref_args), 'incorrect number of args in ctx'
    for i, (name, var) in enumerate(ref_args):
      assert args[i] is var, f'args[{i}] should be {name}'

  def testShape(self):
    x = torch.randn(4, 5)
    y = relu(x)
    expected = x.shape
    dbg = f'got: {y.shape}. expected: {expected}.'
    assert y.shape == expected, f'output shape is wrong. {dbg}'

  def testForward(self):
    x = torch.randn(4, 5)
    y = relu(x)
    ref = F.relu(x)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testForwardNoBatch(self):
    x = torch.randn(1, 5)
    y = relu(x)
    ref = F.relu(x)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testForwardArbitraryShape(self):
    x = torch.randn(4, 5, 6)
    y = relu(x)
    ref = F.relu(x)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testBackward(self):
    x = torch.tensor([[-0.1420527697, -0.4568780065,  0.3526812494],
                      [-0.6894646287,  0.8027195930, -0.3314484954],
                      [-1.2110447884,  0.1189425215,  0.7028403282],
                      [ 1.5301368237,  0.7122527957, -0.3572599888]])
    y = torch.tensor([[0.0000000000, 0.0000000000, 0.3526812494],
                      [0.0000000000, 0.8027195930, 0.0000000000],
                      [0.0000000000, 0.1189425215, 0.7028403282],
                      [1.5301368237, 0.7122527957, 0.0000000000]])
    y.grad = torch.tensor([[-0.5411487818, -0.1987477392,  0.5449076891],
                           [-1.3369221687, -0.8893659711,  0.1755677909],
                           [-2.6910853386,  1.8899576664, -0.3385780454],
                           [-0.2058145702,  0.2141284496, -0.9093903303]])
    ref_x_grad = torch.tensor([[ 0.0000000000,  0.0000000000,  0.5449076891],
                               [ 0.0000000000, -0.8893659711,  0.0000000000],
                               [ 0.0000000000,  1.8899576664, -0.3385780454],
                               [-0.2058145702,  0.2141284496,  0.0000000000]])
    for t in (x,):
      t.grad = torch.zeros_like(t)
    relu_backward(y, x)
    to_test = [
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')
  
  def testBackwardNoBatch(self):
    x = torch.tensor([[ 0.1706864238,  1.7117135525, -0.4802016616,  0.1937024593,  0.7853193879]])
    y = torch.tensor([[0.1706864238, 1.7117135525, 0.0000000000, 0.1937024593, 0.7853193879]])
    y.grad = torch.tensor([[-0.2887229621,  2.3431572914, -0.5021535158, -0.2612630129,  1.1927206516]])
    ref_x_grad = torch.tensor([[-0.2887229621,  2.3431572914,  0.0000000000, -0.2612630129,  1.1927206516]])
    for t in (x,):
      t.grad = torch.zeros_like(t)
    relu_backward(y, x)
    to_test = [
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')


class TestSoftmax(unittest.TestCase):

  def testContext(self):
    ctx = [None]
    x = torch.randn(4, 5)
    y = softmax(x, ctx=ctx)
    # test append to ctx
    assert len(ctx) == 2 and ctx[-1] is not None, 'backward call was not appended to ctx'
    lst_ctx = ctx[-1]
    # test backward_fn, args
    assert len(lst_ctx) == 2, '(backwad_fn, args) should be added ctx'
    backwad_fn, args = lst_ctx
    # test backward_fn
    assert backwad_fn is softmax_backward, 'wrong backward_fn was added to ctx'
    # test args
    ref_args = [('y', y), ('x', x)]
    assert len(args) == len(ref_args), 'incorrect number of args in ctx'
    for i, (name, var) in enumerate(ref_args):
      assert args[i] is var, f'args[{i}] should be {name}'

  def testShape(self):
    x = torch.randn(4, 5)
    y = softmax(x)
    expected = x.shape
    dbg = f'got: {y.shape}. expected: {expected}.'
    assert y.shape == expected, f'output shape is wrong. {dbg}'

  def testForward(self):
    x = torch.randn(4, 5)
    y = softmax(x)
    ref = torch.softmax(x, dim=1)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testForwardStability(self):
    x = torch.randn(4, 5)
    x[0, :] += 100
    x[1, :] += 200
    x[2, :] += 300
    x[3, :] += 400
    y = softmax(x)
    ref = torch.softmax(x, dim=1)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testForwardNoBatch(self):
    x = torch.randn(1, 5)
    y = softmax(x)
    ref = torch.softmax(x, dim=1)
    dbg = f'got: {y}\nexpected: {ref}'
    torch.testing.assert_allclose(y, ref, msg=f'wrong output:\n{dbg}')

  def testBackward(self):
    x = torch.tensor([[-0.0788207650,  1.1206221581, -1.2067351341],
                      [-0.0537703335,  0.8164311647, -0.7430807948],
                      [-0.1624763161,  0.4067649543, -1.5597556829],
                      [ 0.1233877391, -1.3471457958,  1.6308329105]])
    y = torch.tensor([[0.2154255360, 0.7148395777, 0.0697348937],
                      [0.2571147680, 0.6138336658, 0.1290515661],
                      [0.3317635655, 0.5862016678, 0.0820347667],
                      [0.1740649045, 0.0400006101, 0.7859345078]])
    y.grad = torch.tensor([[-0.0418766886, -0.9347261786, -0.7058330178],
                           [-0.4348670244,  0.5590645075,  2.7117042542],
                           [-0.7392839193, -0.7447485328, -0.2368720770],
                           [ 0.6349369884,  1.0240619183, -0.3728161454]])
    ref_x_grad = torch.tensor([[ 0.1474684924, -0.1489042640,  0.0014357530],
                               [-0.2612745166, -0.0136560202,  0.2749305069],
                               [-0.0126109421, -0.0254859626,  0.0380969420],
                               [ 0.1351549029,  0.0466242172, -0.1817791164]])
    for t in (x,):
      t.grad = torch.zeros_like(t)
    softmax_backward(y, x)
    to_test = [
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')
  
  def testBackwardNoBatch(self):
    x = torch.tensor([[-0.0602573790,  0.9555248618,  1.9374154806]])
    y = torch.tensor([[0.0898197964, 0.2480394095, 0.6621408463]])
    y.grad = torch.tensor([[-0.7614974976, -0.6422637701,  0.7784762383]])
    ref_x_grad = torch.tensor([[-0.0942437947, -0.2306817025,  0.3249254823]])
    for t in (x,):
      t.grad = torch.zeros_like(t)
    softmax_backward(y, x)
    to_test = [
      ('x', x, ref_x_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, msg=f'wrong gradient of {name}.\n{dbg}')


class TestCrossEntropy(unittest.TestCase):

  def testContext(self):
    ctx = [None]
    x = torch.randn(4, 5)
    pred = torch.softmax(x, dim=1)
    target = torch.randint(low=0, high=x.size(1), size=(x.size(0),), dtype=torch.long)
    loss = cross_entropy(pred, target, ctx=ctx)
    # test append to ctx
    assert len(ctx) == 2 and ctx[-1] is not None, 'backward call was not appended to ctx'
    lst_ctx = ctx[-1]
    # test backward_fn, args
    assert len(lst_ctx) == 2, '(backwad_fn, args) should be added ctx'
    backwad_fn, args = lst_ctx
    # test backward_fn
    assert backwad_fn is cross_entropy_backward, 'wrong backward_fn was added to ctx'
    # test args
    ref_args = [('loss', loss), ('pred', pred), ('x', target)]
    assert len(args) == len(ref_args), 'incorrect number of args in ctx'
    for i, (name, var) in enumerate(ref_args):
      assert args[i] is var, f'args[{i}] should be {name}'

  def testShape(self):
    x = torch.randn(4, 5)
    pred = torch.softmax(x, dim=1)
    target = torch.randint(low=0, high=x.size(1), size=(x.size(0),), dtype=torch.long)
    loss = cross_entropy(pred, target)
    expected = target.size()
    dbg = f'got: {loss.shape}. expected: {expected}.'
    assert loss.shape == expected, f'output shape is wrong. {dbg}'

  def testForward(self):
    x = torch.randn(4, 5)
    pred = torch.softmax(x, dim=1)
    target = torch.randint(low=0, high=x.size(1), size=(x.size(0),), dtype=torch.long)
    loss = cross_entropy(pred, target)
    ref = torch.nn.functional.cross_entropy(x, target, reduction='none')
    dbg = f'got: {loss}\nexpected: {ref}'
    torch.testing.assert_allclose(loss, ref, atol=1e-3, rtol=1e-3, msg=f'wrong output:\n{dbg}')

  def testForwardStability(self):
    for _ in range(50):
      x = 50 + torch.randn(4, 5)
      x[:, 0] = 0
      pred = torch.softmax(x, dim=1)
      target = torch.randint(low=0, high=x.size(1), size=(x.size(0),), dtype=torch.long)
      loss = cross_entropy(pred, target)
      ref = torch.nn.functional.cross_entropy(x, target, reduction='none')
      dbg = f'got: {loss}\nexpected: {ref}'
      torch.testing.assert_allclose(loss, ref, atol=1e-3, rtol=1e-3, msg=f'wrong output:\n{dbg}')

  def testForwardNoBatch(self):
    x = torch.randn(1, 5)
    pred = torch.softmax(x, dim=1)
    target = torch.randint(low=0, high=x.size(1), size=(x.size(0),), dtype=torch.long)
    loss = cross_entropy(pred, target)
    ref = torch.nn.functional.cross_entropy(x, target, reduction='none')
    dbg = f'got: {loss}\nexpected: {ref}'
    torch.testing.assert_allclose(loss, ref, atol=1e-3, rtol=1e-3, msg=f'wrong output:\n{dbg}')

  def testBackward(self):
    pred = torch.tensor([[0.3046356142, 0.3801231086, 0.2883406281, 0.0269005969],
                         [0.1691603661, 0.0079706665, 0.1763143241, 0.6465547085],
                         [0.2905102968, 0.0904196799, 0.1445289403, 0.4745410383]])
    target = torch.tensor([2, 2, 0])
    y = torch.tensor([1.2436127663, 1.7354869843, 1.2361162901])
    y.grad = torch.tensor([-1.0623387098, -0.0725658908, -0.4252309203])
    ref_pred_grad = torch.tensor([[0.0000000000, 0.0000000000, 3.6843185425, 0.0000000000],
                                  [0.0000000000, 0.0000000000, 0.4115711749, 0.0000000000],
                                  [1.4637378454, 0.0000000000, 0.0000000000, 0.0000000000]])
    for t in (pred,):
      t.grad = torch.zeros_like(t)
    cross_entropy_backward(y, pred, target)
    to_test = [
      ('pred', pred, ref_pred_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, atol=1e-3, rtol=1e-3, msg=f'wrong gradient of {name}.\n{dbg}')

  def testBackwardNoBatch(self):
    pred = torch.tensor([[0.5238355994, 0.3022331297, 0.0616454147, 0.1122858971]])
    target = torch.tensor([0])
    y = torch.tensor([0.6465773582])
    y.grad = torch.tensor([-0.7106007934])
    ref_pred_grad = torch.tensor([[1.3565340042, 0.0000000000, 0.0000000000, 0.0000000000]])
    for t in (pred,):
      t.grad = torch.zeros_like(t)
    cross_entropy_backward(y, pred, target)
    to_test = [
      ('pred', pred, ref_pred_grad),
    ]
    for name, var, ref_grad in to_test:
      dbg = f'got: {var.grad}\nexpected: {ref_grad}'
      torch.testing.assert_allclose(var.grad, ref_grad, atol=1e-3, rtol=1e-3, msg=f'wrong gradient of {name}.\n{dbg}')
