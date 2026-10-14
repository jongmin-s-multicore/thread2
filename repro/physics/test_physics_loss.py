"""Check temporal discretization and an independently finite-differenced gradient."""
import unittest
import numpy as np
import torch
from physics_loss_pilot import transition

class PhysicsLossTests(unittest.TestCase):
 def test_constant_force_closed_form_and_gradient(self):
  dt=.01;mass=4.188790204786391;a=mass/(mass+dt);b=100*dt/(mass+dt);p=dict(Ah=(np.eye(2)*a).tolist(),Bh=(np.eye(2)*b).tolist(),dt=dt)
  s=torch.tensor([[.2,.3,-.1,.4]],dtype=torch.double,requires_grad=True);u=torch.tensor([[.2,-.1]],dtype=torch.double)
  got=transition(s,u,p);v=s.detach().numpy()[0,2:];force=u.numpy()[0];power=a**25;sum_power=a*(1-power)/(1-a)
  expected=np.r_[s.detach().numpy()[0,:2]+dt*(sum_power*v+b/(1-a)*(25-sum_power)*force),power*v+b*(1-power)/(1-a)*force]
  np.testing.assert_allclose(got.detach().numpy()[0],expected,atol=1e-12)
  val=got.square().sum();grad=torch.autograd.grad(val,s)[0].numpy()[0]
  numerical=[]
  for i in range(4):
   plus=s.detach().clone();minus=s.detach().clone();plus[0,i]+=1e-6;minus[0,i]-=1e-6;numerical.append(float((transition(plus,u,p).square().sum()-transition(minus,u,p).square().sum())/2e-6))
  np.testing.assert_allclose(grad,numerical,rtol=1e-6,atol=1e-8)
 def test_rest_is_fixed_point(self):
  s=torch.tensor([[1.,2.,0.,0.]],dtype=torch.double);p=dict(Ah=(np.eye(2)*.9976).tolist(),Bh=(np.eye(2)*.238).tolist(),dt=.01)
  self.assertTrue(torch.equal(transition(s,torch.zeros(1,2,dtype=torch.double),p),s))

if __name__=='__main__':unittest.main()
