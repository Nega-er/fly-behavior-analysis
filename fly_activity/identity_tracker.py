"""Reserved identities: missing observations never become arbitrary free slots."""
import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self, config):
        self.c = config
        self.n = config['expected_flies']
        self.pos = np.full((self.n, 2), np.nan)
        self.velocity = np.zeros((self.n, 2))
        self.last_frame = np.full(self.n, -1)
        self.segment = np.zeros(self.n, dtype=int)
        self.appearance = [None] * self.n
        self.pending = {}

    def stationary(self, i):
        return np.linalg.norm(self.velocity[i]) <= self.c['move_start_speed']

    def update(self, detections, frame, fps, measurement_dt_s=None, image_updated=True, descriptors=None):
        dt = measurement_dt_s or 1/fps
        points = np.asarray(detections, float).reshape(-1, 2)
        appearance = [None]*len(points) if descriptors is None else list(descriptors)
        close = np.zeros(len(points), bool)
        if len(points) > 1:
            distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            close = distances.min(axis=1) < self.c['collision_distance_px']
        appearance = [a for a, reject in zip(appearance, close) if not reject]
        points = points[~close]
        known = np.flatnonzero(self.last_frame >= 0)
        assigned, used = {}, set()
        costs = np.full((len(known), len(points)), 1e9)
        for r, i in enumerate(known):
            age = (frame-self.last_frame[i])/fps
            continuous = self.last_frame[i] == frame-1
            stopped = self.stationary(i)
            hold = self.c.get('stationary_identity_hold_s', 120) if stopped else self.c.get('identity_hold_s', 3)
            if not continuous and age > hold:
                continue
            elapsed = dt if continuous else age
            prediction = self.pos[i]+self.velocity[i]*(elapsed if image_updated else 0) if continuous else self.pos[i]
            gate = self.c['max_speed_px_s']*dt if continuous else min(
                self.c.get('identity_recovery_radius_px', 80),
                max(self.c.get('stationary_recovery_radius_px', 24) if stopped else 0,
                    self.c['max_speed_px_s']*age))
            for j, point in enumerate(points):
                if np.linalg.norm(point-self.pos[i]) > gate:
                    continue
                penalty = 0.
                if appearance[j] is not None and self.appearance[i] is not None:
                    delta = np.abs(np.asarray(appearance[j])-self.appearance[i])
                    # Reject gross size/brightness changes rather than relabel another fly.
                    if delta[0] > np.log(2.5) or delta[2] > .3:
                        continue
                    penalty = self.c.get('identity_appearance_weight', 20)*float(np.linalg.norm(delta))
                costs[r, j] = np.linalg.norm(point-prediction)+penalty
        if costs.size:
            # A stale position must not steal a well-supported continuous track.
            for continuous in (True, False):
                group=[r for r,i in enumerate(known) if (self.last_frame[i]==frame-1)==continuous]
                available=[j for j in range(len(points)) if j not in used]
                if not group or not available:
                    continue
                local=costs[np.ix_(group,available)]
                rr,jj=linear_sum_assignment(local)
                for r,j in zip(rr,jj):
                    value=local[r,j]
                    margin=self.c['assignment_margin_px']
                    if value>=1e9 or np.any(np.delete(local[r],j)<value+margin) or np.any(np.delete(local[:,j],r)<value+margin):
                        continue
                    i=int(known[group[r]])
                    candidate=available[j]
                    # Distant returns need evidence over multiple images. Close
                    # stationary returns are already supported by their anchor.
                    distance=np.linalg.norm(points[candidate]-self.pos[i])
                    if not continuous and distance>self.c.get('stationary_recovery_radius_px',24):
                        previous=self.pending.get(i)
                        count=1
                        if previous is not None and previous[0]==frame-1 and np.linalg.norm(points[candidate]-previous[1])<=max(20,self.c['max_speed_px_s']*dt):
                            count=previous[2]+int(image_updated)
                        self.pending[i]=(frame,points[candidate].copy(),count)
                        if count<3:
                            continue
                    self.pending.pop(i,None)
                    assigned[i]=candidate
                    used.add(candidate)
        # A known but missing ID is reserved for the entire video. Only unused
        # identities may start a track; candidates competing for known IDs cannot
        # bypass an ambiguous assignment by becoming a new identity.
        remaining = [j for j in range(len(points)) if j not in used and not np.any(costs[:, j] < 1e9)]
        remaining.sort(key=lambda j: (points[j, 0], points[j, 1]))
        for i, j in zip(np.flatnonzero(self.last_frame < 0), remaining):
            assigned[int(i)] = j
        rows = []
        for i in range(self.n):
            base = dict(frame=frame, time_s=frame/fps, slot=i+1)
            if i in assigned:
                j = assigned[i]
                q = points[j]
                seen = self.last_frame[i] >= 0
                continuous = seen and self.last_frame[i] == frame-1
                gap = (frame-self.last_frame[i])/fps if seen and not continuous else 0.
                match_distance = float(np.linalg.norm(q-self.pos[i])) if seen else ''
                old_x,old_y = self.pos[i] if seen else ('','')
                if continuous and image_updated:
                    self.velocity[i] = .5*self.velocity[i]+.5*(q-self.pos[i])/dt
                elif not continuous:
                    self.segment[i] += 1
                    self.velocity[i] = 0
                if appearance[j] is not None:
                    a = np.asarray(appearance[j], float)
                    self.appearance[i] = a if self.appearance[i] is None else .9*self.appearance[i]+.1*a
                self.pos[i], self.last_frame[i] = q, frame
                base.update(segment=int(self.segment[i]), x_px=float(q[0]), y_px=float(q[1]), status='observed',
                            identity_state='tracked' if continuous else 'reacquired' if seen else 'initialized', identity_gap_s=gap,
                            identity_match_distance_px=match_distance,identity_previous_x_px=old_x,identity_previous_y_px=old_y)
            else:
                base.update(segment=0, x_px='', y_px='', status='unknown',
                            identity_state='missing_reserved' if self.last_frame[i]>=0 else 'uninitialized',
                            identity_gap_s=(frame-self.last_frame[i])/fps if self.last_frame[i]>=0 else '',
                            identity_match_distance_px='',identity_previous_x_px='',identity_previous_y_px='')
            rows.append(base)
        return rows, int(close.sum()), len(points)
