from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, polygonize
from trajectories_clustering import TrajectoryClustering
from trajectory_visualizer_matplot import TrajectoryVisualizerMatplot
import numpy as np
import geopandas as gpd
from distance_measures import FrechetDist


class PlotSimilarity:

    @staticmethod
    def plot_area_between(traj0, traj1, switch_xy=True, axis=None):
        if switch_xy:
            traj0 = [[lon, lat] for lat, lon in traj0]
            traj1 = [[lon, lat] for lat, lon in traj1]
        _polygon_points = traj0 + traj1[::-1] + traj0[0:1]
        _polygon = Polygon(_polygon_points)
        x, y = _polygon.exterior.xy
        ls = LineString(np.c_[x, y])
        lr = LineString(ls.coords[:] + ls.coords[0:1])
        mls = unary_union(lr)
        for _polygon in polygonize(mls):
            p = gpd.GeoSeries(_polygon)
            p.plot(ax=axis, color="lightgreen")
        axis.text(_polygon.centroid.x, _polygon.centroid.y, "Area between two trajectories", fontsize=8, color='black',
                  horizontalalignment='center', verticalalignment='bottom')

    @staticmethod
    def plot_frechet_dist(traj_xy0, traj_xy1,
                          traj0, traj1, axis=None):
        f_d, p_i, q_i = FrechetDist.frechetdist_index(traj_xy0, traj_xy1)
        x = [traj0[p_i][1], traj1[q_i][1]]
        y = [traj0[p_i][0], traj1[q_i][0]]
        axis.plot(x, y, label="frechet dist", linestyle=":", color="purple")
        axis.text(np.mean(x), np.mean(y), "Frechet Dist", fontsize=8, color='black',
                  horizontalalignment='center', verticalalignment='bottom')

