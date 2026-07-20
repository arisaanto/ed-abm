| Metric | Baseline | COCPIT only | NURSTA only | Both |
| --- | --- | --- | --- | --- |
| **Connectivity (visible points)** | 50.9 | 81.6 | 50.9 | 81.2 |
| **Isovist area (m²)** | 103.1 | 158.4 | 104.3 | 159.0 |
| **Through vision (m)** | 307.0 | 559.9 | 304.7 | 553.5 |
| **Visual integration (0–1)** | 0.404 | 0.534 | 0.404 | 0.532 |
| **Visual intelligibility (r)** | 0.763 | 0.953 | 0.766 | 0.953 |

*Note.* VGA-style approximations on a 1.0 m regular grid over the largest connected walkable component. Connectivity is the mean count of directly visible grid points. Isovist area is the mean 360-degree ray-cast visible area. Through vision is the mean cumulative Euclidean length of direct visibility links per grid point. Visual integration is mean normalized closeness centrality on the visibility graph. Visual intelligibility is the Pearson correlation between local connectivity and global visual integration across grid points. Condition-specific visibility walls are used throughout; these are not formal DepthmapX outputs. Sample points: Baseline n=250, COCPIT only n=250, NURSTA only n=253, Both n=253.
