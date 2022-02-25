import unittest
import torch

from model import softmax, cross_entropy, softmax_classifier, softmax_classifier_backward

##########################################################
# Softmax
##########################################################

class Softmax(unittest.TestCase):

  def testShape(self):
    x = torch.randn(size=(5, 7))
    out = softmax(x)
    debug = f'x: {x}\nout: {out}'
    self.assertTrue(out.shape == x.shape, msg='softmax output should have the same shape as input')

  def testIsProbability(self):
    x = torch.randn(size=(5, 7))
    out = softmax(x)
    debug = f'x: {x}\nout: {out}'
    self.assertTrue(torch.all(out >= 0), msg=f'softmax output should be non-negative.\n{debug}')
    torch.testing.assert_allclose(out.sum(dim=1), torch.ones(size=(5,)), msg=f'softmax output rows should sum to 1.\n{debug}')

  def testNoBatch(self):
    x = torch.randn(size=(1, 7))
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    debug = f'x: {x}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'softmax output is incorrect (batch_size=1).\n{debug}')

  def testBatch(self):
    x = torch.randn(size=(5, 7))
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    debug = f'x: {x}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'softmax output is incorrect (batch_size>1).\n{debug}')

  def testNumericalStability(self):
    x = torch.randn(size=(5, 7)) + 500
    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    debug = f'x: {x}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'softmax output is not numerically stable.\n{debug}')


##########################################################
# Cross Entropy
##########################################################

class CrossEntropy(unittest.TestCase):
  def setUp(self):
    self.eps = 1e-7

  def testShape(self):
    pred = torch.rand(size=(5, 7))
    target = torch.randint(low=0, high=pred.size(1), size=(pred.size(0),), dtype=torch.long)
    pred /= pred.sum(dim=1, keepdims=True)
    log_pred = pred.clamp(self.eps, 1 - self.eps).log()
    out = cross_entropy(pred, target)
    debug = f'pred: {pred}\ntarget: {target}\nout: {out}'
    self.assertTrue(out.shape == (), msg='cross_entropy output should be a scalar')

  def testNoBatch(self):
    pred = torch.rand(size=(1, 7))
    target = torch.randint(low=0, high=pred.size(1), size=(pred.size(0),), dtype=torch.long)
    pred /= pred.sum(dim=1, keepdims=True)
    log_pred = pred.clamp(self.eps, 1 - self.eps).log()
    out = cross_entropy(pred, target)
    ref = torch.nn.functional.nll_loss(log_pred, target)
    debug = f'pred: {pred}\ntarget: {target}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'cross_entropy output is incorrect (batch_size=1).\n{debug}')

  def testBatch(self):
    pred = torch.rand(size=(5, 7))
    target = torch.randint(low=0, high=pred.size(1), size=(pred.size(0),), dtype=torch.long)
    pred /= pred.sum(dim=1, keepdims=True)
    log_pred = pred.clamp(self.eps, 1 - self.eps).log()
    out = cross_entropy(pred, target)
    ref = torch.nn.functional.nll_loss(log_pred, target)
    debug = f'pred: {pred}\ntarget: {target}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'cross_entropy output is incorrect (batch_size>1).\n{debug}')

  def testNumericalStability(self):
    pred = torch.rand(size=(1, 7))
    pred[:, 0] = 0
    target = torch.randint(low=0, high=pred.size(1), size=(pred.size(0),), dtype=torch.long)
    pred /= pred.sum(dim=1, keepdims=True)
    log_pred = pred.clamp(self.eps, 1 - self.eps).log()
    out = cross_entropy(pred, target)
    ref = torch.nn.functional.nll_loss(log_pred, target)
    debug = f'pred: {pred}\ntarget: {target}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'cross_entropy output is not numerically stable.\n{debug}')


##########################################################
# Softmax Classifier
##########################################################

class SoftmaxClassifier(unittest.TestCase):
  def setUp(self):
    self.w = torch.tensor([[ 0., -1.,  0.,  1.],
                           [ 0., -1., -1.,  0.],
                           [-1., -1.,  1.,  0.],
                           [-1., -1., -1., -1.],
                           [ 0.,  0., -1.,  1.],
                           [ 0.,  1.,  1., -1.]])
    self.b = torch.tensor([ 1.,  0., -1., -1., -1.,  0.])

  def tearDown(self):
    del self.w
    del self.b
  
  def testNoBatch(self):
    x = torch.tensor([[0.0418225, 0.2236919, 0.0220730, 0.9709660]])
    out = softmax_classifier(x, self.w, self.b)
    ref = torch.tensor([[0.6874346, 0.0936840, 0.0345446, 0.0125176, 0.1138163, 0.0580028]])
    debug = f'x: {x}\nw: {self.w}\nb: {self.b}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'softmax_classifier output is incorrect (batch_size=1).\n{debug}')

  def testBatch(self):
    x = torch.tensor([[0.0418225, 0.2236919, 0.0220730, 0.9709660],
                      [0.1071336, 0.3263596, 0.2006417, 0.7961680],
                      [0.6126359, 0.3509384, 0.9443940, 0.4181272],
                      [0.2187163, 0.0626919, 0.4096224, 0.1317171],
                      [0.0716444, 0.3564701, 0.1463558, 0.4978001],
                      [0.6608164, 0.8039287, 0.1025592, 0.4446338],
                      [0.6281751, 0.7725115, 0.6889389, 0.6429738],
                      [0.3036749, 0.9889252, 0.9259661, 0.0302742],
                      [0.9099082, 0.4162390, 0.2444494, 0.1980238],
                      [0.3814440, 0.6388829, 0.9266685, 0.5952389]])
    out = softmax_classifier(x, self.w, self.b)
    ref = torch.tensor([[0.6874346, 0.0936840, 0.0345446, 0.0125176, 0.1138163, 0.0580028],
                        [0.6442470, 0.0874680, 0.0431819, 0.0130393, 0.0988694, 0.1131944],
                        [0.4689521, 0.0441671, 0.0582143, 0.0057963, 0.0350594, 0.3878106],
                        [0.5021369, 0.1075046, 0.0721006, 0.0278574, 0.0480356, 0.2423650],
                        [0.5524811, 0.1067258, 0.0489760, 0.0222161, 0.0922529, 0.1773482],
                        [0.4170931, 0.0887760, 0.0207059, 0.0108121, 0.1138269, 0.3487859],
                        [0.4387482, 0.0426068, 0.0331723, 0.0043967, 0.0645538, 0.4165222],
                        [0.1268303, 0.0179324, 0.0310273, 0.0047240, 0.0182800, 0.8012059],
                        [0.4525677, 0.1069610, 0.0258279, 0.0129946, 0.0727280, 0.3289209],
                        [0.4281355, 0.0343820, 0.0551152, 0.0047629, 0.0434519, 0.4341526]])
    debug = f'x: {x}\nw: {self.w}\nb: {self.b}\nout: {out}\nref: {ref}'
    torch.testing.assert_allclose(out, ref, msg=f'softmax_classifier output is incorrect (batch_size>1).\n{debug}')


##########################################################
# Softmax Classifier Backward
##########################################################

class SoftmaxClassifierBackward(unittest.TestCase):
  def setUp(self):
    self.w = torch.tensor([[ 0.4552463591,  1.8834323883,  0.1890305728, -1.9750952721],
                           [-0.8811249733, -1.4499745369, -0.1827174425, -1.1477377415],
                           [ 0.0333099402, -1.7030651569,  0.7322748303, -1.0514358282],
                           [ 0.8741850257, -1.1595503092, -1.4959537983, -0.4884476066],
                           [ 1.9152492285,  1.0828249454,  1.5929585695,  0.0641875863]])
    self.b = torch.tensor([-0.7941429615, -0.7943550944, -0.6679479480,  0.2889612019, 0.6793746948])

  def tearDown(self):
    del self.w
    del self.b
  
  def testNoBatch(self):
    x = torch.tensor([[ 0.3817181289,  0.4734393358, -1.6752171516, -2.1894714832]])
    pred = torch.tensor([[ 6.3071376085e-01, 2.3798899725e-02, 5.9394617565e-03, 3.3594477177e-01, 3.6031340715e-03]])
    target = torch.tensor([3], dtype=torch.long)
    w_grad_ref = torch.tensor([[ 2.4075487256e-01,  2.9860469699e-01, -1.0565824509e+00, -1.3809298277e+00],
                               [ 9.0844687074e-03,  1.1267331429e-02, -3.9868313819e-02, -5.2106995136e-02],
                               [ 2.2672002669e-03,  2.8119748458e-03, -9.9498881027e-03, -1.3004282489e-02],
                               [-2.5348192453e-01, -3.1438985467e-01,  1.1124366522e+00,  1.4539300203e+00],
                               [ 1.3753816020e-03,  1.7058653757e-03, -6.0360319912e-03, -7.8889597207e-03]])
    b_grad_ref = torch.tensor([ 0.6307137609,  0.0237988923,  0.0059394618, -0.6640552282, 0.0036031341])
    softmax_classifier_backward(x, self.w, self.b, pred, target)
    debug = f'x: {x}\ny: {pred}\ntarget: {target}\nw: {self.w}\nb: {self.b}\nw.grad: {self.w.grad}\nw.grad (ref): {w_grad_ref}\nb.grad: {self.b.grad}\nb.grad (ref): {b_grad_ref}'
    torch.testing.assert_allclose(self.w.grad, w_grad_ref, msg=f'softmax_classifier_backward w.grad is incorrect (batch_size=1).\n{debug}')
    torch.testing.assert_allclose(self.b.grad, b_grad_ref, msg=f'softmax_classifier_backward b.grad is incorrect (batch_size=1).\n{debug}')

  def testBatch(self):
    x = torch.tensor([[ 0.3817181289,  0.4734393358, -1.6752171516, -2.1894714832],
                      [ 2.9298193455,  0.3587905765,  0.2316469997, -0.0883881971],
                      [-0.8051397204, -1.8023374081,  1.5362951756, -1.2212771177],
                      [-0.2489842474, -0.3497180939, -1.0500792265,  1.5737550259],
                      [ 1.8822822571,  0.3370933831, -0.9906531572, -0.5440846086],
                      [-1.0704393387, -0.2831149697,  0.2842271030,  0.1917557716],
                      [ 1.2218270302,  0.1589236110,  0.8741209507, -1.6024698019],
                      [ 0.7916751504, -0.2085955590, -0.9030384421, -0.0943523049],
                      [-1.2771739960,  0.4536714852,  0.5619912744,  0.1021014750],
                      [-0.4908930361,  0.1570422947, -0.4170274734,  2.0359976292]])
    pred = torch.tensor([[ 6.3071376085e-01, 2.3798899725e-02, 5.9394617565e-03, 3.3594477177e-01, 3.6031340715e-03],
                         [ 3.6242513452e-03, 1.8627255486e-05, 3.4474313725e-04, 7.2764977813e-03, 9.8873585463e-01],
                         [ 9.8139932379e-04, 2.4065248668e-01, 7.4826335907e-01, 6.0937488452e-03, 4.0089064278e-03],
                         [ 1.8925389741e-03, 4.5998919755e-02, 2.0218467340e-02, 8.8881248236e-01, 4.3077539653e-02],
                         [ 9.2116363347e-02, 2.2296926472e-03, 4.9795741215e-03, 5.0741088390e-01, 3.9326354861e-01],
                         [ 3.9394240826e-02, 4.4614276290e-01, 2.6998996735e-01, 1.4488559961e-01, 9.9587380886e-02],
                         [ 2.3813517392e-01, 5.2555613220e-03, 3.3389650285e-02, 1.5317918733e-02, 7.0790171623e-01],
                         [ 2.6616342366e-02, 2.3958301172e-02, 2.5638408959e-02, 8.2252788544e-01, 1.0125906765e-01],
                         [ 2.4310554564e-01, 2.6069363952e-01, 1.3853053749e-01, 4.7744795680e-02, 3.0992555618e-01],
                         [ 7.0930300280e-03, 5.0955761224e-02, 2.9465572909e-02, 4.4061973691e-01, 4.7186592221e-01]])
    target = torch.tensor([3, 3, 2, 0, 3, 3, 3, 1, 3, 1], dtype=torch.long)
    w_grad_ref = torch.tensor([[ 0.0628377497,  0.0810788423,  0.0231586993, -0.3340643048],
                               [-0.1302803159, -0.0390445329,  0.1834425926, -0.2011248320],
                               [-0.0209006965,  0.0442252681, -0.0274438225,  0.0393498242],
                               [-0.2952070832, -0.1607219279, -0.2118963301,  0.5337232947],
                               [ 0.3835503161,  0.0744623914,  0.0327388607, -0.0378839560]])
    b_grad_ref = torch.tensor([ 0.0283672698, -0.0900295228,  0.0276759751, -0.2783365846, 0.3123228550])
    softmax_classifier_backward(x, self.w, self.b, pred, target)
    debug = f'x: {x}\ny: {pred}\ntarget: {target}\nw: {self.w}\nb: {self.b}\nw.grad: {self.w.grad}\nw.grad (ref): {w_grad_ref}\nb.grad: {self.b.grad}\nb.grad (ref): {b_grad_ref}'
    torch.testing.assert_allclose(self.w.grad, w_grad_ref, msg=f'softmax_classifier_backward w.grad is incorrect (batch_size>1).\n{debug}')
    torch.testing.assert_allclose(self.b.grad, b_grad_ref, msg=f'softmax_classifier_backward b.grad is incorrect (batch_size>1).\n{debug}')
