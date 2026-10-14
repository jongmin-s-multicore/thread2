"""Tests catch wrong action-block intervention and unequal diagonal magnitudes."""
import unittest
import numpy as np
from intervention import grid,branch_actions,analytic
from analyze_intervention import aggregate,energy_terms


class InterventionTests(unittest.TestCase):
    def test_grid_symmetry_and_magnitude(self):
        conditions=grid();self.assertEqual(len(conditions),41)
        for start in [1,9,17,25,33]:
            for i in range(4):
                a,b=conditions[start+i],conditions[start+i+4]
                np.testing.assert_allclose(np.array(a['control'])+b['control'],0,atol=1e-15)
                self.assertAlmostEqual(np.linalg.norm(a['control']),a['radius'])

    def test_only_last_block_changes_and_roundtrip(self):
        hist=np.arange(20).reshape(10,2)/30;mean=np.array([.1,-.2]);std=np.array([.3,.5])
        zero=branch_actions(hist,np.zeros((5,2)),mean,std)
        changed=branch_actions(hist,np.tile([.2,-.1],(5,1)),mean,std)
        np.testing.assert_array_equal(zero[:2],changed[:2]);self.assertFalse(np.array_equal(zero[2],changed[2]))
        np.testing.assert_allclose(changed[2].reshape(5,2)*std+mean,np.tile([.2,-.1],(5,1)))

    def test_damping_and_action_response(self):
        a=.9976183615964902;b=.23816384035097887
        params=dict(Ah=(np.eye(2)*a).tolist(),Bh=(np.eye(2)*b).tolist(),dt=.01)
        st=[1.,1.,1.,-.5];z,cl=analytic(st,np.zeros((5,2)),params)
        v,cc=analytic(st,np.tile([.1,0],(5,1)),params)
        self.assertEqual(cl+cc,0)
        np.testing.assert_allclose(z[-1,2:],a**25*np.array(st[2:]))
        np.testing.assert_allclose(v[-1,2:]-z[-1,2:],[.1*b*sum(a**i for i in range(25)),0],atol=1e-14)

    def test_episode_weighting_and_energy_sign(self):
        rows=[dict(episode=1,error=0.),dict(episode=1,error=2.),dict(episode=2,error=5.)]
        result=aggregate(rows,'error');self.assertEqual(result['mean'],3.)
        r=energy_terms(np.array([0.,0.,2.,0.]),np.array([.25,0.,1.,0.]),np.zeros(2),np.eye(2),np.eye(2))
        self.assertEqual(r['delta_k'],-1.5);self.assertEqual(r['necessary_excess'],-1.5)


if __name__=='__main__':unittest.main()
