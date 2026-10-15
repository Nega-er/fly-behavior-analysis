import unittest
import numpy as np
import cv2
from analyze_flies import DEFAULTS
from video_setup import foreground
from movement_rate import summarize_rates


class BodyAndRatesTests(unittest.TestCase):
    def test_bright_body_survives_area_limit_shadow_rejected(self):
        gray=np.zeros((160,250),np.uint8)
        cv2.ellipse(gray,(75,80),(27,12),0,0,360,85,-1)  # wings / broad faint skirt
        cv2.ellipse(gray,(75,80),(15,7),0,0,360,240,-1)  # actual body
        cv2.ellipse(gray,(160,95),(12,6),0,0,360,50,-1) # separate faint shadow
        bg=np.zeros_like(gray);config=dict(DEFAULTS,max_area_px=500)
        _,old=foreground(gray,bg,dict(config,body_detection_mode="background_diff"))
        n,labels,stats,points=cv2.connectedComponentsWithStats(old)
        old_points=[points[i] for i in range(1,n) if 40<=stats[i,4]<=500]
        self.assertEqual(len(old_points),1)
        self.assertGreater(old_points[0][0],140)
        _,mask,info=foreground(gray,bg,config,True)
        n,labels,stats,points=cv2.connectedComponentsWithStats(mask)
        detected=[points[i] for i in range(1,n) if 40<=stats[i,4]<=500]
        self.assertEqual(len(detected),1)
        self.assertLess(np.linalg.norm(detected[0]-[75,80]),1)
        self.assertEqual(info["body_core_applied"],1)

    def test_dim_video_auto_mode_preserves_original_mask(self):
        gray=np.zeros((120,160),np.uint8);cv2.ellipse(gray,(80,60),(12,6),0,0,360,110,-1)
        _,expected=foreground(gray,np.zeros_like(gray),dict(DEFAULTS,body_detection_mode="background_diff"))
        _,actual,info=foreground(gray,np.zeros_like(gray),DEFAULTS,True)
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(info["body_core_applied"],0)

    def test_dark_body_core(self):
        gray=np.full((120,160),240,np.uint8);cv2.ellipse(gray,(80,60),(12,6),0,0,360,15,-1)
        _,mask,info=foreground(gray,np.full_like(gray,240),dict(DEFAULTS,foreground_polarity="dark"),True)
        self.assertEqual(mask[60,80],1)
        self.assertEqual(mask[10,10],0)
        self.assertEqual(info["body_core_applied"],1)

    def test_time_rate_uses_actual_window_and_valid_time(self):
        bins=[dict(start_s=0,end_s=1,slot=1,distance=10,valid_s=1,unit="mm"),
              dict(start_s=0,end_s=1,slot=2,distance=5,valid_s=.5,unit="mm"),
              dict(start_s=1,end_s=1.5,slot=1,distance=5,valid_s=.5,unit="mm"),
              dict(start_s=1,end_s=1.5,slot=2,distance=0,valid_s=0,unit="mm")]
        rates=summarize_rates(bins,2)
        self.assertAlmostEqual(rates[0]["distance_per_wall_s"],15)
        self.assertAlmostEqual(rates[0]["mean_distance_per_valid_fly_second"],10)
        self.assertAlmostEqual(rates[0]["interval_coverage"],.75)
        self.assertAlmostEqual(rates[1]["distance_per_wall_s"],10)
        self.assertAlmostEqual(rates[1]["interval_coverage"],.5)
        self.assertAlmostEqual(sum(r["measured_distance"] for r in rates),20)

    def test_no_observation_is_gap_not_zero(self):
        rates=summarize_rates([dict(start_s=0,end_s=1,slot=1,distance=0,valid_s=0,unit="px")],1)
        self.assertEqual(rates[0]["distance_per_wall_s"],"")
        self.assertEqual(rates[0]["mean_distance_per_valid_fly_second"],"")
        self.assertEqual(rates[0]["interval_coverage"],0)


if __name__=="__main__":unittest.main()
