import unittest
import cv2
import numpy as np
from analyze_flies import DEFAULTS, Tracker
from video_setup import temporal_background, foreground, recover_stationary, estimate_population


class IdentityTests(unittest.TestCase):
    def tracker(self, **kw):
        return Tracker(dict(DEFAULTS,expected_flies=2,collision_distance_px=5,**kw))

    def test_missing_stationary_id_returns_to_its_position_not_detection_order(self):
        t=self.tracker()
        t.update([(20,0),(100,0)],0,10)
        for frame in range(1,30):
            rows,_,_=t.update([(100,0)],frame,10)
            self.assertEqual(rows[0]['identity_state'],'missing_reserved')
            self.assertEqual(rows[1]['x_px'],100)
        rows,_,_=t.update([(100,0),(21,0)],30,10)
        self.assertEqual(rows[0]['x_px'],21)
        self.assertEqual(rows[0]['identity_state'],'reacquired')
        self.assertEqual(rows[0]['segment'],2)
        self.assertEqual(rows[1]['segment'],1)

    def test_outsider_cannot_steal_reserved_identity(self):
        t=self.tracker()
        t.update([(20,0),(100,0)],0,10)
        rows,_,_=t.update([(100,0),(300,0)],1,10)
        self.assertEqual(rows[0]['status'],'unknown')
        rows,_,_=t.update([(100,0),(300,0)],2,10)
        self.assertEqual(rows[0]['status'],'unknown')
        self.assertEqual(t.pos[0,0],20)

    def test_stale_id_does_not_block_continuous_neighbor(self):
        t=self.tracker(assignment_margin_px=4)
        t.update([(20,0),(40,0)],0,10)
        t.update([(40,0)],1,10)
        rows,_,_=t.update([(38,0)],2,10)
        self.assertEqual(rows[1]['status'],'observed')
        self.assertEqual(rows[1]['x_px'],38)
        self.assertEqual(rows[0]['status'],'unknown')

    def test_distant_reappearance_needs_multiple_images(self):
        t=self.tracker()
        t.update([(20,0)],0,10)
        t.update([],1,10)
        for k in (2,3):
            rows,_,_=t.update([(80,0)],k,10)
            self.assertEqual(rows[0]['status'],'unknown')
        rows,_,_=t.update([(80,0)],4,10)
        self.assertEqual(rows[0]['identity_state'],'reacquired')
        self.assertEqual(rows[0]['segment'],2)

    def test_ambiguity_cannot_create_new_id(self):
        t=self.tracker(assignment_margin_px=10)
        t.update([(20,0)],0,10)
        rows,_,_=t.update([(16,0),(24,0)],1,10)
        self.assertTrue(all(r['status']=='unknown' for r in rows))

    def test_appearance_rejects_incompatible_replacement(self):
        t=self.tracker()
        a=[np.log(100),np.log(2),.9,.1]
        t.update([(20,0)],0,10,descriptors=[a])
        t.update([],1,10,descriptors=[])
        rows,_,_=t.update([(20,0)],2,10,descriptors=[[np.log(400),np.log(2),.4,.1]])
        self.assertEqual(rows[0]['status'],'unknown')

    def test_temporal_background_keeps_long_stops_but_not_static_artifacts(self):
        empty=np.zeros((80,120),np.uint8)
        cv2.rectangle(empty,(5,5),(20,10),230,-1)
        body=empty.copy();cv2.ellipse(body,(60,40),(12,6),0,0,360,240,-1)
        samples=[body]*8+[empty]*3
        median=np.median(samples,axis=0).astype(np.uint8)
        bg=temporal_background(samples,DEFAULTS)
        _,old=foreground(body,median,DEFAULTS)
        _,mask=foreground(body,bg,DEFAULTS)
        self.assertEqual(old[40,60],0)
        self.assertEqual(mask[40,60],1)
        self.assertEqual(mask[7,10],0)

    def test_static_rescue_requires_known_anchor_and_image_evidence(self):
        gray=np.zeros((100,200),np.uint8)
        cv2.ellipse(gray,(60,50),(12,6),0,0,360,240,-1)
        cv2.ellipse(gray,(160,50),(12,6),0,0,360,240,-1)
        t=self.tracker();t.update([(60,50)],0,10)
        info=dict(body_core_applied=1,body_core_threshold=168)
        rescued,pixels=recover_stationary(gray,np.zeros_like(gray),t,1,10,info)
        self.assertGreater(pixels,40)
        self.assertEqual(rescued[50,60],1)
        self.assertEqual(rescued[50,160],0)
        absent,pixels=recover_stationary(np.zeros_like(gray),np.zeros_like(gray),t,2,10,info)
        self.assertEqual(pixels,0)

    def test_supported_upper_count_avoids_contact_undercount_and_single_spike(self):
        result=estimate_population([5]*20+[7]*10+[12])
        self.assertEqual(result['estimated_flies'],7)
        self.assertEqual(result['modal_count'],5)
        self.assertEqual(estimate_population([2,3,3,3,4])['estimated_flies'],3)


if __name__=='__main__':unittest.main()
