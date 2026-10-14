import unittest,numpy as np
from physics_depth import background_map,match_background,diagnose
from intervention import analytic
class Tests(unittest.TestCase):
 def test_background_map_excludes_green_objects(self):
  b=np.ones((6,20,20,3),dtype=np.uint8)*50;a=b+3;a[:,10,10]=[40,140,20]
  correction,mask=background_map(a,b)
  np.testing.assert_array_equal(correction[0,0],[3,3,3]);self.assertTrue(mask[10,10]);np.testing.assert_array_equal(correction[10,10],[0,0,0]);self.assertEqual(int(match_background(b,correction)[0,0,0,0]),53)
 def test_free_energy_sign_and_clipping(self):
  mass=4.188790204786391;aa=1/(1+.01/mass);params=dict(M=(np.eye(2)*mass).tolist(),D=np.eye(2).tolist(),G=(100*np.eye(2)).tolist(),Ah=(aa*np.eye(2)).tolist(),Bh=(.01*100/mass*aa*np.eye(2)).tolist(),dt=.01,walls=[dict(center=[100.,100.],halfsize=[1.,1.])],particle_radius=.1)
  tr,cl=analytic([0,0,1,0],np.zeros((5,2)),params);d=diagnose(tr,[0,0],params);self.assertTrue(d['free']);self.assertLess(abs(d['fine_energy_residual']),1e-12)
  tr,cl=analytic([0,0,6,0],np.zeros((5,2)),params);d=diagnose(tr,[0,0],params,clips=cl);self.assertFalse(d['free']);self.assertLess(abs(d['fine_energy_residual']),1e-12)
if __name__=='__main__':unittest.main()
