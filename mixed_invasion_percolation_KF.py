import heapq as hq
import logging
import numpy as np
from collections import namedtuple

from openpnm.algorithms import Algorithm
from openpnm.topotools import find_clusters, site_percolation
from openpnm.utils import Docorator

logger = logging.getLogger(__name__)
docstr = Docorator()


class MIPSettings:
    # phase = ''
    pore_volume = 'pore.volume'
    throat_volume = 'throat.volume'
    pore_entry_pressure = "pore.entry_pressure"
    throat_entry_pressure = "throat.entry_pressure"
    snap_off = ""
    invade_isolated_Ts = False
    late_pore_filling = ""
    late_throat_filling = ""


class MixedInvasionPercolation(Algorithm):
    r"""
    Mixed invasion percolation algorithm allowing pores and throats
    to invade separately.
    """

    def __init__(self, phase, name="mip_?", **kwargs):
        super().__init__(name=name, **kwargs)
        self.settings._update(MIPSettings())
        # Store Phase object directly
        self.settings["phase"] = phase
        self.reset()

    def setup(self,
        phase=None,
        pore_entry_pressure="pore.entry_pressure",
        throat_entry_pressure="throat.entry_pressure",
        snap_off="",
        invade_isolated_Ts=False,
        late_pore_filling="",
        late_throat_filling="",
    ):
        # phase = self.settings["phase"]
        # # pore entry pressures
        # self.settings["pore_entry_pressure"] = pore_entry_pressure
        # self["pore.entry_pressure"] = phase[pore_entry_pressure]
        # # throat entry pressures
        # self.settings["throat_entry_pressure"] = throat_entry_pressure
        # self["throat.entry_pressure"] = phase[throat_entry_pressure]
        # # other flags
        # self.settings["snap_off"] = snap_off
        # self.settings["invade_isolated_Ts"] = invade_isolated_Ts
        # self.settings["late_pore_filling"] = late_pore_filling
        # self.settings["late_throat_filling"] = late_throat_filling
        # self.reset()

        if phase:
            self.settings["phase"] = phase
        if throat_entry_pressure:
            self.settings["throat_entry_pressure"] = throat_entry_pressure
        self["throat.entry_pressure"] = phase[self.settings["throat_entry_pressure"]]
        if len(np.shape(self["throat.entry_pressure"])) > 1:
            self._bidirectional = True
        else:
            self._bidirectional = False
        if pore_entry_pressure:
            self.settings["pore_entry_pressure"] = pore_entry_pressure
        self["pore.entry_pressure"] = phase[self.settings["pore_entry_pressure"]]
        if snap_off:
            self.settings["snap_off"] = snap_off
        if invade_isolated_Ts:
            self.settings["invade_isolated_Ts"] = invade_isolated_Ts
        if late_pore_filling:
            self.settings["late_pore_filling"] = late_pore_filling
        if late_throat_filling:
            self.settings["late_throat_filling"] = late_throat_filling
        self.reset()

    def reset(self):
        self["pore.invasion_pressure"] = np.inf
        self["throat.invasion_pressure"] = np.inf
        self["pore.invasion_sequence"] = -1
        self["throat.invasion_sequence"] = -1
        self["pore.cluster"] = -1
        self["throat.cluster"] = -1
        self["pore.trapped"] = np.inf
        self["throat.trapped"] = np.inf
        self["pore.inlets"] = False
        self["pore.outlets"] = False
        self._interface_Ts = np.zeros(self.Nt, dtype=bool)
        self._interface_Ps = np.zeros(self.Np, dtype=bool)
        self.queue = []

    def set_inlet_BC(self, pores=None, clusters=None):
        if pores is not None:
            logger.info("Setting inlet pores at shared pressure")
            clusters = []
            clusters.append(pores)
        elif clusters is not None:
            logger.info("Setting inlet clusters at individual pressures")
        else:
            logger.error("Either 'inlets' or 'clusters' must be passed to" + " setup method")
        self.queue = []
        for i, cluster in enumerate(clusters):
            self.queue.append([])
            # Perform initial analysis on input pores
            self["pore.invasion_sequence"][cluster] = 0
            self["pore.cluster"][cluster] = i
            self["pore.invasion_pressure"][cluster] = -np.inf
            if np.size(cluster) > 1:
                for elem_id in cluster:
                    self._add_ts2q(elem_id, self.queue[i])

            elif np.size(cluster) == 1:
                self._add_ts2q(cluster, self.queue[i])
            else:
                logger.warning("Some inlet clusters have no pores")
        if self.settings["snap_off"]:
            self._apply_snap_off()

    def _add_ts2q(self, pore, queue):
        """
        Helper method to add throats to the cluster queue
        """
        net = self.project.network
        elem_type = "throat"
        # Find throats connected to newly invaded pore
        Ts = net.find_neighbor_throats(pores=pore)
        # Remove already invaded throats from Ts
        Ts = Ts[self["throat.invasion_sequence"][Ts] <= 0]
        tcp = self["throat.entry_pressure"]
        if len(Ts) > 0:
            self._interface_Ts[Ts] = True
            for T in Ts:
                data = []
                # Pc
                if self._bidirectional:
                    # Get index of pore being invaded next and apply correct
                    # entry pressure
                    pmap = net["throat.conns"][T] != pore
                    pind = list(pmap).index(True)
                    data.append(tcp[T][pind])
                else:
                    data.append(tcp[T])
                # Element Index
                data.append(T)
                # Element Type (Pore of Throat)
                data.append(elem_type)
                hq.heappush(queue, data)

    def _add_ps2q(self, throat, queue):
        """
        Helper method to add pores to the cluster queue
        """
        net = self.project.network
        elem_type = "pore"
        # Find pores connected to newly invaded throat
        Ps = net["throat.conns"][throat]
        # Remove already invaded pores from Ps
        Ps = Ps[self["pore.invasion_sequence"][Ps] <= 0]
        if len(Ps) > 0:
            self._interface_Ps[Ps] = True
            for P in Ps:
                data = []
                # Pc
                data.append(self["pore.entry_pressure"][P])
                # Element Index
                data.append(P)
                # Element Type (Pore of Throat)
                data.append(elem_type)
                hq.heappush(queue, data)

    def run(self, max_pressure=np.inf):

        if "throat.entry_pressure" not in self.keys():
            logger.error("Setup method must be run first")

        if max_pressure is None:
            self.max_pressure = np.inf
        else:
            self.max_pressure = max_pressure
        if len(self.queue) == 0:
            raise RuntimeError("No inlets set")
        
        # track whether each cluster has reached the maximum pressure
        self.max_p_reached = [False] * len(self.queue)
        # starting invasion sequence
        self.count = 0
        # highest pressure reached so far - used for porosimetry curve
        self.high_Pc = np.ones(len(self.queue)) * -np.inf
        outlets = self["pore.outlets"]      
        terminate_clusters = np.sum(outlets) > 0
        if not hasattr(self, "invasion_running"):
            self.invasion_running = [True] * len(self.queue)
        else:
            # created by set_residual
            pass

        while np.any(self.invasion_running) and not np.all(self.max_p_reached):
            # Loop over clusters
            for c_num in np.argwhere(self.invasion_running).flatten():
                self._invade_cluster(c_num)
                queue = self.queue[c_num]
                if len(queue) == 0 or self.max_p_reached[c_num]:
                    # If the cluster contains no more entries invasion has
                    # finished
                    self.invasion_running[c_num] = False
            if self.settings["invade_isolated_Ts"]:
                self._invade_isolated_Ts()
            if terminate_clusters:
                # terminated clusters
                tcs = np.unique(self["pore.cluster"][outlets]).astype(int)
                tcs = tcs[tcs >= 0]
                if len(tcs) > 0:
                    for tc in tcs:
                        if self.invasion_running[tc] is True:
                            self.invasion_running[tc] = False
                            logger.info(
                                "Cluster "
                                + str(tc)
                                + " reached "
                                + " outlet at sequence "
                                + str(self.count)
                            )

    def _invade_cluster(self, c_num):
        queue = self.queue[c_num]
        pressure, elem_id, elem_type = hq.heappop(queue)
        if elem_type == "pore":
            self._interface_Ps[elem_id] = False
        else:
            self._interface_Ts[elem_id] = False
        if pressure > self.max_pressure:
            self.max_p_reached[c_num] = True
        else:
            elem_cluster = self[elem_type + ".cluster"][elem_id]
            elem_cluster = elem_cluster.astype(int)
            # Cluster is the uninvaded cluster
            if elem_cluster == -1:
                self.count += 1
                # Record highest Pc cluster has reached
                if self.high_Pc[c_num] < pressure:
                    self.high_Pc[c_num] = pressure
                # The newly invaded element is available for
                # invasion
                self[elem_type + ".invasion_sequence"][elem_id] = self.count
                self[elem_type + ".cluster"][elem_id] = c_num
                self[elem_type + ".invasion_pressure"][elem_id] = self.high_Pc[c_num]
                if elem_type == "throat":
                    self._add_ps2q(elem_id, queue)
                elif elem_type == "pore":
                    self._add_ts2q(elem_id, queue)

            # Element is part of existing cluster that is still invading
            elif elem_cluster != c_num and self.invasion_running[elem_cluster]:
                # The newly invaded element is part of an invading
                # cluster. Merge the clusters using the existing
                # cluster number)
                self._merge_cluster(c2keep=c_num, c2empty=elem_cluster)
                logger.info(
                    "Merging cluster "
                    + str(elem_cluster)
                    + " into cluster "
                    + str(c_num)
                    + " at sequence "
                    + str(self.count)
                )
            # Element is part of residual cluster - now invasion can start
            elif elem_cluster != c_num and len(self.queue[elem_cluster]) > 0:
                # The newly invaded element is part of an invading
                # cluster. Merge the clusters using the existing
                # cluster number)
                self._merge_cluster(c2keep=c_num, c2empty=elem_cluster)
                logger.info(
                    "Merging residual cluster "
                    + str(elem_cluster)
                    + " into cluster "
                    + str(c_num)
                    + " at sequence "
                    + str(self.count)
                )
            else:
                pass
            
    def _merge_cluster(self, c2keep, c2empty):
        r"""
        Little helper function to merger clusters but only add the uninvaded
        elements
        """
        while len(self.queue[c2empty]) > 0:
            temp = [_pc, _id, _type] = hq.heappop(self.queue[c2empty])
            if self[_type + ".invasion_sequence"][_id] == -1:
                hq.heappush(self.queue[c2keep], temp)
        self.invasion_running[c2empty] = False

    def results(self, Pc):
        r"""
        Places the results of the IP simulation into the Phase object.

        Parameters
        ----------
        Pc : float
            Capillary Pressure at which phase configuration was reached

        """
        if Pc is None:
            results = {
                "pore.invasion_sequence": self["pore.invasion_sequence"],
                "throat.invasion_sequence": self["throat.invasion_sequence"],
            }
        else:
            phase = self.settings['phase']
            net = self.project.network
            inv_p = self["pore.invasion_pressure"].copy()
            inv_t = self["throat.invasion_pressure"].copy()
            # Handle trapped pores and throats by moving their pressure up
            # to be ignored
            if np.sum(self["pore.invasion_sequence"] == -1) > 0:
                inv_p[self["pore.invasion_sequence"] == -1] = Pc + 1
            if np.sum(self["throat.invasion_sequence"] == -1) > 0:
                inv_t[self["throat.invasion_sequence"] == -1] = Pc + 1
            p_inv = inv_p <= Pc
            t_inv = inv_t <= Pc

            if self.settings["late_pore_filling"]:
                # Set pressure on phase to current capillary pressure
                phase["pore.pressure"] = Pc
                # Regenerate corresponding physics model
                for phys in self.project.find_physics(phase=phase):
                    phys.regenerate_models(self.settings["late_pore_filling"])
                # Fetch partial filling fraction from phase object (0->1)
                frac = phase[self.settings["late_pore_filling"]]
                p_vol = net["pore.volume"] * frac
            else:
                p_vol = net["pore.volume"]
            if self.settings["late_throat_filling"]:
                # Set pressure on phase to current capillary pressure
                phase["throat.pressure"] = Pc
                # Regenerate corresponding physics model
                for phys in self.project.find_physics(phase=phase):
                    phys.regenerate_models(self.settings["late_throat_filling"])
                # Fetch partial filling fraction from phase object (0->1)
                frac = phase[self.settings["late_throat_filling"]]
                t_vol = net["throat.volume"] * frac
            else:
                t_vol = net["throat.volume"]
            results = {
                "pore.occupancy": p_inv * p_vol,
                "throat.occupancy": t_inv * t_vol,
            }
        return results

    def pc_curve(self,inv_points=None):
        r"""
        Get the percolation data as the non-wetting phase saturation vs the
        capillary pressure.

        """
        net = self.project.network
        pvols = net[self.settings['pore_volume']]
        tvols = net[self.settings['throat_volume']]
        tot_vol = np.sum(pvols) + np.sum(tvols)
        # Normalize
        pvols /= tot_vol
        tvols /= tot_vol
        # Remove trapped volume

        if "pore.invasion_pressure" not in self.props():
            logger.error("You must run the algorithm first.")
            return None

        if inv_points is None:
            mask = ~np.isnan(self["throat.invasion_pressure"])
            ok_Pc = self["throat.invasion_pressure"][mask]
            inv_points = np.unique(ok_Pc)
        sat_p = np.zeros(len(inv_points))
        sat_t = np.zeros(len(inv_points))

        for i, Pc in enumerate(inv_points):
            res = self.results(Pc=Pc)
            sat_p[i] = np.sum(res["pore.occupancy"])
            sat_t[i] = np.sum(res["throat.occupancy"])

        sat_p /= tot_vol
        sat_t /= tot_vol
        tot_sat = sat_p + sat_t
        tot_sat /= tot_vol

        pc_curve = namedtuple("pc_curve", ("pc", "S_pore", "S_throat", "S_tot"))
        data = pc_curve(inv_points, sat_p, sat_t, tot_sat)
        return data