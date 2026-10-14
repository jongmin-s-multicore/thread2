import unittest
import numpy as np
from zero_velocity import conditions,coherent_trace,decompose
class Tests(unittest.TestCase):
 def test_history_and_future_solve(self):
  A=np.eye(2)*.9976183615964902;B=np.eye(2)*.23816384035097887;p=dict(Ah=A,Bh=B,dt=.01)
  q=np.array([2.,2.]);v=np.array([.25,-.1]);u=np.array([.05,0.]);t=coherent_trace(q,v,u,p)
  np.testing.assert_allclose(t[51],np.r_[q,v],atol=1e-14)
  for n in range(76):
   np.testing.assert_allclose(t[n+1,2:],A@t[n,2:]+(B@u if n<51 else 0),atol=1e-14)
   np.testing.assert_allclose(t[n+1,:2],t[n,:2]+.01*t[n+1,2:],atol=1e-14)
  np.testing.assert_allclose(t[-1,2:],np.linalg.matrix_power(A,25)@v,atol=1e-14)
 def test_pairs_decompose_cross_axis_and_bias(self):
  b=np.array([.1,-.2]);J=np.array([[.9,.1],[-.03,.95]]);r=.25
  fp=np.array([b+J[:,j]*r for j in range(2)]);fm=np.array([b-J[:,j]*r for j in range(2)])
  actual,err=decompose(b,fp,fm,r,.94213);np.testing.assert_allclose(actual,J);np.testing.assert_allclose(err,J-.94213*np.eye(2),atol=1e-15)
 def test_grid_paired_histories(self):
  rows=conditions();self.assertEqual(len(rows),23)
  for h in ['push_positive','push_negative']:
   paired=[x for x in rows if x['history']==h];self.assertEqual(len(paired),5)
   for x in paired:self.assertTrue(any(y['history']=='coast' and y['velocity']==x['velocity'] for y in rows))
if __name__=='__main__':unittest.main()

class StatisticsTests(unittest.TestCase):
 def test_site_weighting_and_signed_bias(self):
  from analyze_zero_velocity import grouped,estimate
  result=grouped([dict(episode=1,error=0.),dict(episode=1,error=2.),dict(episode=2,error=5.)],'error')
  self.assertEqual(result['mean'],3.)
  self.assertEqual(estimate([[1.,-1.],[3.,-3.]])['mean'],[2.,-2.])
