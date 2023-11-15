import os
import pickle
import re
import sys
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import traci.constants as tc
import randomTrips  # noqa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import xml.etree.ElementTree as ET
import pickle
fontfamily = ["Helvetica", "Arial"]
fontsize = 10


matplotlib.rc('xtick', direction='in')
matplotlib.rc('ytick', direction='in')
matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": fontfamily, "size": fontsize})
matplotlib.rc('savefig', bbox='tight', format='svg', pad_inches=0.05)
matplotlib.rc('legend', fontsize='small')

# import seaborn as sns
# sns.set_style("whitegrid")

# sumoBinary_gui = "C:\Program Files (x86)\\Eclipse\Sumo\\bin\sumo-gui"
# sumoBinary = "C:\Program Files (x86)\\Eclipse\Sumo\\bin\sumo"
# base_path = "C:\\Users\Fan Wu\OneDrive\WithYou\pedestrian_coordinate\sumo\my_crossing\\"

sumoBinary_gui = "C:\\Users\cheng\Documents\sumo-1.8.0\\bin\sumo-gui"
sumoBinary = "C:\\Users\cheng\Documents\sumo-1.8.0\\bin\sumo"
base_path = "C:\\Users\cheng\OneDrive\Documents\Withyou\pedestrian_coordinate\sumo\my_crossing\\"

net = base_path + "co_arterial.net.xml"
# sumoCmd_gui = [sumoBinary_gui, "-c", base_path+"crossing_cfg.sumocfg",
#            "--tripinfo-output", base_path+"trip_output.xml"]
# sumoCmd = [sumoBinary, "-c", base_path+"crossing_cfg.sumocfg",
#            "--tripinfo-output", base_path+"trip_output.xml"]
sumoCmd_gui = [sumoBinary_gui, "--tripinfo-output", base_path + "trip_output.xml",
               "--duration-log.statistics"]
sumoCmd = [sumoBinary, "--tripinfo-output", base_path + "trip_output.xml",
           "--duration-log.statistics"]
outputfile = base_path + "pedestrians.trip.xml"

id_sets = {'WE': {'03', '04', '05', '08', '10', '13', '14'},
           'EW': {'16', '17', '18', '21', '23', '26', '27'}}

lane_sets = {'WE': {'gneE14_0': 1, 'gneE14_1': 2, 'gneE15_0': 3, 'gneE15_1': 4, 'gneE16_0': 5, 'gneE16_1': 6},
             'EW': {'gneE19_0': 1, 'gneE19_1': 2, 'gneE20_0': 3, 'gneE20_1': 4, 'gneE21_0': 5, 'gneE21_1': 6}}


def generate_passenger_demand(pph, seed=0):
    """
    :param pph: int, passenger per hour
    """
    period = str(round(3600 / pph))
    randomTrips.main(randomTrips.get_options([
        '--net-file', net,
        '--output-trip-file', outputfile,
        '--seed', str(seed),  # make runs reproducible
        '--pedestrians',
        '--prefix', 'ped',
        # prevent trips that start and end on the same edge
        '--min-distance', '1',
        '--trip-attributes', 'departPos="random" arrivalPos="random"',
        '--binomial', '1',
        '--period', period,
        '-e', '7200']))


# minimum green time for the vehicles
MIN_GREEN_TIME = 15
# the first phase in tls plan. see 'pedcrossing.tll.xml'
VEHICLE_GREEN_PHASE = 0
PEDESTRIAN_GREEN_PHASE = 2
WALKINGAREAS = [':T02_w0', ':T02_w1']
CROSSINGS = [':T02_c0']
TLSID = 'T02'


# def unit_run(param, signal_control_type='proposed', pph=30, speeddev=0.1, gui=False):
#     if gui:
#         sumo = sumoCmd_gui.copy()
#     else:
#         sumo = sumoCmd.copy()
#     sumo = sumo + ['--default.speeddev', str(speeddev)]
#
#     if signal_control_type == 'unsignalized':
#         sumo = sumo + ['--route-files', base_path + 'vehicle.rou.xml']
#     else:
#         generate_passenger_demand(pph)
#         sumo = sumo + ['--route-files', base_path + 'vehicle.rou.xml' + ', ' + base_path + 'pedestrians.trip.xml']
#
#     traci.start(sumo)
#     return run(param, signal_control_type)

def unit_run(param, signal_control_type='synchronized', crossing_control_type='proposed', pph=30, speeddev=0.1,
             gui=False, seed=0, trajectory=False, vph_ratio=1.0, forecast_models=None, filter_edge=True):
    """

    :param filter_edge:
    :param forecast_models:
    :param crossing_control_type:
    :param param:
    :param signal_control_type:
    :param pph:
    :param speeddev:
    :param gui:
    :param seed:
    :param trajectory: False or a string representing the file name of output trajectory
    :return: loop_results, light_results, ped_wait
    """
    if gui:
        sumo = sumoCmd_gui.copy()
    else:
        sumo = sumoCmd.copy()
    sumo = sumo + ['--default.speeddev', str(speeddev)]
    generate_passenger_demand(pph, seed=seed)
    sumo = sumo + ['--seed', str(seed)]

    if signal_control_type == 'unsignalized':
        sumo = sumo + ['-c', base_path + "crossing_cfg_unsignalized.sumocfg"]
    elif signal_control_type == 'fixed_sync':
        sumo = sumo + ['-c', base_path + "crossing_cfg_fixed_sync.sumocfg"]
    elif signal_control_type == 'fixed_unsync':
        sumo = sumo + ['-c', base_path + "crossing_cfg_fixed_unsync.sumocfg"]
    elif signal_control_type == 'synchronized':
        sumo = sumo + ['-c', base_path + "crossing_cfg_synchronized.sumocfg"]
    elif signal_control_type == 'unsynchronized':
        sumo = sumo + ['-c', base_path + "crossing_cfg_unsynchronized.sumocfg"]

    vph_ratio = str(vph_ratio)
    vph_file = base_path + "vehicle_{}.rou.xml, " + base_path + "pedestrians.trip.xml"
    sumo = sumo + ['-r', vph_file.format(vph_ratio)]

    if trajectory:
        sumo = sumo + ['--fcd-output', base_path + trajectory]
        if filter_edge:
            sumo = sumo + ['--fcd-output.filter-edges.input-file', base_path + "selected_edges.filter"]
    traci.start(sumo)
    return run(param, crossing_control_type, forecast_models)


def run(param, control_type='proposed', forecast_models=None):
    """
    execute the TraCI control loop

    :param control_type: one of {'Pelican', 'proposed', 'proposed_cv'}
    :param param: A dict like dict(delta_t_min=5, delta_t_max=40, gamma=0, v=13.5)
    :return: loop_results, light_results, ped_wait
    """
    # track the duration for which the green phase of the vehicles has been
    # active
    greenTimeSoFar = 0
    band_overlap_list = []
    # whether the pedestrian button has been pressed
    activeRequest = False
    cumPedWaitTime = 0
    q1 = param['q1']
    q3 = param['q3']
    v = param['v']
    delta_t_max = param['delta_t_max']
    ped_extra_wait = 0  # Time from button pushed to greenTimeSoFar > MIN_GREEN_TIME:
    duration = param['duration']

    L1 = traci.lane.getLength('gneE17_0')
    L3 = traci.lane.getLength('gneE18_0')

    # Subscribe loop detector data
    loop_id_list = traci.inductionloop.getIDList()
    light_id_list = traci.trafficlight.getIDList()
    loop_results = {loop_id: [] for loop_id in loop_id_list}
    light_results = {light_id: [] for light_id in light_id_list}

    ped_wait = []
    # main loop. do something every simulation step until no more vehicles are
    # loaded or running
    while traci.simulation.getMinExpectedNumber() > 0:

        # Subscribe loop detectors and traffic light data for each step
        for loop_id in loop_id_list:
            traci.inductionloop.subscribe(loop_id, [tc.LAST_STEP_VEHICLE_DATA])
        for light_id in light_id_list:
            traci.trafficlight.subscribe(light_id, [tc.TL_CURRENT_PHASE])
        traci.simulationStep()
        loop_result = traci.inductionloop.getAllSubscriptionResults()
        light_result = traci.trafficlight.getAllSubscriptionResults()
        for light_id, result in light_result.items():
            light_results[light_id].append(result[40])
        for loop_id, result in loop_result.items():
            loop_results[loop_id].extend(result[23])

        # decide whether there is a waiting pedestrian and switch if the green
        # phase for the vehicles exceeds its minimum duration
        if control_type == "unsignalized":
            continue

        if control_type == 'fixed':
            continue

        if not activeRequest:
            activeRequest, numWaiting = checkWaitingPersons()
        if traci.trafficlight.getPhase(TLSID) == VEHICLE_GREEN_PHASE:
            greenTimeSoFar += 1
            # check whether someone has pushed the button
            if activeRequest:
                if greenTimeSoFar > MIN_GREEN_TIME:
                    if cumPedWaitTime == 0:
                        if control_type == "Pelican":
                            pedWaitTime = 1
                        elif control_type == "proposed":
                            pedWaitTime, band_overlap = getWaitingTime2(L1, L3, q1, q3, v, duration=duration, delta_t_max=delta_t_max-ped_extra_wait)
                            band_overlap_list.append(band_overlap.copy())
                        elif control_type == 'proposed_cv':
                            pedWaitTime = getWaitingTime3(L1, L3, v, duration=duration,
                                                                        delta_t_max=delta_t_max - ped_extra_wait,
                                                                        forecast_models=forecast_models)
                            if pedWaitTime>10:  # when the forecast could be inaccurate. Re-estimate in the next time
                                ped_extra_wait += 1
                                continue
                        else:
                            raise ValueError('Wrong control type')
                        ped_wait.append((traci.simulation.getTime()-ped_extra_wait, numWaiting, pedWaitTime+ped_extra_wait))
                        cumPedWaitTime += 1
                        ped_extra_wait = 0
                    elif cumPedWaitTime < pedWaitTime:
                        cumPedWaitTime += 1
                        # continue
                    elif cumPedWaitTime == pedWaitTime:
                        # switch to the next phase
                        traci.trafficlight.setPhase(
                            TLSID, VEHICLE_GREEN_PHASE + 1)
                        # reset state
                        activeRequest = False
                        greenTimeSoFar = 0
                        cumPedWaitTime = 0

                # If someone has pushed the button but greenTimeSoFar <= MIN_GREEN_TIME
                else:
                    ped_extra_wait += 1

    # sys.stdout.flush()
    traci.close()
    return loop_results, light_results, ped_wait, band_overlap_list


def get_traffic_light_param(ID):
    """
    :param ID: The ID of traffic light
    :return: phase_duration_list, cycle length
    """
    a = traci.trafficlight.getAllProgramLogics(ID)
    phase_duration_list = [i.duration for i in a[1].phases]
    return phase_duration_list, np.sum(phase_duration_list)


def get_current_traffic_light_state(ID, phase_duration_list, C):
    """Calculate the t_{g1}, t_{g2}, t_{r1}, t_{r2} for current simulation time"""
    current_phase = traci.trafficlight.getPhase(ID)
    next_switch_time = traci.trafficlight.getNextSwitch(ID) - traci.simulation.getTime()
    start_to_now = np.sum(phase_duration_list[0:current_phase + 1]) - next_switch_time
    start_to_red = np.sum(phase_duration_list[0:2])
    start_to_green = np.sum(phase_duration_list[0:4])
    tg1 = (start_to_green - start_to_now) % C
    tr1 = (start_to_red - start_to_now) % C
    tg2 = tg1 + C
    tr2 = tr1 + C
    return tg1, tg2, tr1, tr2


def project_traffic_light_to_crossing(ID, phase_duration_list, C, delta_t):
    """Calculate g_start1, g_end1, g_start2, g_end2 of the intersection ID at the crossing."""
    current_phase = traci.trafficlight.getPhase(ID)
    next_switch_time = traci.trafficlight.getNextSwitch(ID) - traci.simulation.getTime()
    start_to_now = np.sum(phase_duration_list[0:current_phase + 1]) - next_switch_time
    start_to_red = np.sum(phase_duration_list[0:2])
    start_to_green = np.sum(phase_duration_list[0:4])

    g_start = [(start_to_red - start_to_now) % C - delta_t]
    while g_start[-1] < 2*C:
        g_start.append(g_start[-1] + C)
    g_start = np.array(g_start)
    g_start = g_start[g_start>=0]

    g_end = [(start_to_green - start_to_now) % C - delta_t]
    while g_end[-1] < 2*C:
        g_end.append(g_end[-1] + C)
    g_end = np.array(g_end)
    g_end = g_end[g_end>0]

    if g_end[0] < g_start[0]:
        g_start = np.concatenate([[0], g_start])

    return g_start[0:2], g_end[0:2]  # Considering the last a few seconds are often not used



def getWaitingTime2(L1, L3, q1, q3, v, delta_t_max=40, duration=23):
    """
    Get the best waiting time within `delta_t_max` that has the minimum
    q-weighted overlapping band area
    """
    t = traci.simulation.getTime()
    light_param = {'T01': get_traffic_light_param('T01'),
                   'T03': get_traffic_light_param('T03')}
    phase_duration_list1, C1 = light_param['T01']
    phase_duration_list3, C3 = light_param['T03']

    g_start1, g_end1 = project_traffic_light_to_crossing('T01', phase_duration_list1, C1, L1/v)
    g_start3, g_end3 = project_traffic_light_to_crossing('T03', phase_duration_list3, C3, L3/v)

    wait_time_candidate = range(2, delta_t_max + 3)
    band_overlap = [get_band_overlap(g_start1, g_end1, g_start3, g_end3, q1, q3, t, duration) for t in wait_time_candidate]
    best_waiting_time = wait_time_candidate[np.argmin(band_overlap)]
    return best_waiting_time - 1, band_overlap


def getWaitingTime3(L1, L3, v, delta_t_max=40, duration=23, forecast_models=None):
    """
    Get the best waiting time within `delta_t_max` that has the minimum
    q-weighted overlapping band area
    """
    from collections import defaultdict
    t = traci.simulation.getTime()
    light_param = {'T01': get_traffic_light_param('T01'),
                   'T03': get_traffic_light_param('T03')}
    phase_duration_list1, C1 = light_param['T01']
    phase_duration_list3, C3 = light_param['T03']

    g_start1, g_end1 = project_traffic_light_to_crossing('T01', phase_duration_list1, C1, L1/v)
    g_start3, g_end3 = project_traffic_light_to_crossing('T03', phase_duration_list3, C3, L3/v)

    v = v-1
    direction_cycle_length = {'WE':C1, 'EW':C3}
    lane_sets2 = {'WE': {'-gneE17_0', '-gneE17_1'},
                 'EW': {'-gneE18_0', '-gneE18_1'}}
    L_set = {'WE':L1, 'EW':L3}
    time_set = {}

    id_list = traci.vehicle.getIDList()
    veh_info = [[id, traci.vehicle.getLaneID(id), traci.vehicle.getLanePosition(id), traci.vehicle.getSpeed(id)] for id in id_list]
    veh_info = pd.DataFrame(data=veh_info, columns=['id', 'lane', 'pos', 'speed'])
    # Get the number of vehicles ahead
    veh_info = veh_info.sort_values(['lane'])
    veh_info['n_ahead'] = veh_info.groupby(['lane'])['pos'].transform(lambda x: np.argsort(-x).argsort())

    for direction in ['WE', 'EW']:
        light = t % direction_cycle_length[direction]
        upstream_veh_info = veh_info.loc[(veh_info.lane.isin(lane_sets[direction]))&(veh_info.id.apply(lambda x:x[0:2]).isin(id_sets[direction])), :].copy()
        upstream_veh_info['light'] = light
        upstream_veh_info['lane_number'] = upstream_veh_info.lane.map(lane_sets[direction])
        if upstream_veh_info.shape[0]>=1:
            upstream_veh_time = forecast_models[direction].predict(upstream_veh_info.loc[:, ['lane_number', 'light', 'pos', 'speed', 'n_ahead']])
            upstream_veh_time += 100/v
        else:
            upstream_veh_time = np.array([])

        stream_veh_info = veh_info.loc[veh_info.lane.isin(lane_sets2[direction]), :]
        stream_veh_time = (L_set[direction] - stream_veh_info.pos.values)/v
        time_set[direction] = np.concatenate([upstream_veh_time, stream_veh_time])

    wait_time_candidate = range(1, delta_t_max)
    N_overlap = [get_N_overlap(g_start1, g_end1, g_start3, g_end3, time_set, t, duration) for t in wait_time_candidate]
    # N_overlap = N_overlap + np.arange(len(N_overlap)) / 10
    best_waiting_time = wait_time_candidate[np.argmin(N_overlap)]

    return best_waiting_time


def get_N_overlap(g_start1, g_end1, g_start3, g_end3, time_set, t, duration):
    N = np.sum((time_set['EW']<min(g_end1[0], t+duration)) & (time_set['EW']>max(g_start1[0], t)))
    N += np.sum((time_set['EW'] < min(g_end1[1], t + duration)) & (time_set['EW'] > max(g_start1[1], t)))
    N += np.sum((time_set['WE'] < min(g_end3[0], t + duration)) & (time_set['WE'] > max(g_start3[0], t)))
    N += np.sum((time_set['WE'] < min(g_end3[0], t + duration)) & (time_set['WE'] > max(g_start3[0], t)))
    return N


def get_band_overlap(g_start1, g_end1, g_start3, g_end3, q1, q3, t, duration):
    B1 = max(0, min(g_end1[0], t + duration) - max(g_start1[0], t)) * q1
    B3 = max(0, min(g_end3[0], t + duration) - max(g_start3[0], t)) * q3

    B1 += max(0, min(g_end1[1], t + duration) - max(g_start1[1], t)) * q1
    B3 += max(0, min(g_end3[1], t + duration) - max(g_start3[1], t)) * q3

    return B1+B3


def getWaitingTime(light_ID, L, v, delta_t_min, delta_t_max, gamma):
    light_param = {'T01': get_traffic_light_param('T01'),
                   'T03': get_traffic_light_param('T03')}
    phase_duration_list, C = light_param[light_ID]
    tg1, tg2, tr1, tr2 = get_current_traffic_light_state(light_ID, phase_duration_list, C)
    tg = tg1 if (tg1 * v - L - delta_t_min * v) > 0 else tg2
    tr = tr1 if (tr1 * v - L - delta_t_min * v) > 0 else tr2
    if tg < tr:
        tc = np.min([(tg * v - L) / v, delta_t_max])
    else:
        tc = (tr * v - L) / v
        if (tc - delta_t_min) < gamma:
            tc = np.min([(tg * v - L) / v, delta_t_max])
        else:
            tc = delta_t_min
    return tc


def checkWaitingPersons():
    """check whether a person has requested to cross the street"""

    # check both sides of the crossing
    for edge in WALKINGAREAS:
        peds = traci.edge.getLastStepPersonIDs(edge)
        # check who is waiting at the crossing
        # we assume that pedestrians push the button upon
        # standing still for 1s
        for ped in peds:
            if (traci.person.getWaitingTime(ped) == 1 and
                traci.person.getNextEdge(ped) in CROSSINGS and
                traci.trafficlight.getPhase(TLSID) == VEHICLE_GREEN_PHASE):
                numWaiting = traci.trafficlight.getServedPersonCount(TLSID, PEDESTRIAN_GREEN_PHASE)
                print("%s: pedestrian %s pushes the button (waiting: %s)" %
                      (traci.simulation.getTime(), ped, numWaiting))
                return True, numWaiting
    return False, 0


def loop_result_to_dataframe(loop_result):
    data = pd.DataFrame(loop_result,
                        columns=('vID', 'length', 't_start', 't_end', 'type'))
    data = data.loc[data.t_end > 0, ['vID', 't_start', 't_end']]
    return data


def light_result_to_dataframe(light_result, g2y, y2r, r2g):
    """
    :param light_result: list, the output of run(), recording signal phase of every second
    :param g2y: the phase just change green to yellow
    :param y2r: the phase just change yellow to red
    :param r2g: the phase just change red to green
    :return: Dataframe with columns: n_cycle  green_begin  yellow_begin  red_begin  cycle_begin
    """
    n_cylce = 0
    green_begin = np.nan
    yellow_begin = np.nan
    cycle_length = 0
    last_phase = -1
    collect = []
    for t, phase in enumerate(light_result):
        if phase == last_phase:
            pass
        elif phase == g2y:  # if just change to yellow phase
            yellow_begin = t
        elif phase == r2g:  # if just change to green phase
            green_begin = t
        elif phase == y2r:  # if just change to red phase
            red_begin = t
            collect.append((n_cylce, green_begin, yellow_begin, red_begin, cycle_length))
            cycle_length = 0
            n_cylce += 1
        cycle_length += 1
        last_phase = phase

    data = pd.DataFrame(collect,
                        columns=('n_cylce', 'green_begin', 'yellow_begin', 'red_begin', 'cycle_length'))
    data.iloc[0, data.iloc[0, :].values < 0] = np.nan
    return data


def plot_pcd(light_data, loop_data, v=13.5, L=100, et=2):
    # Traffic light part
    cum_cycle = np.cumsum(light_data['cycle_length'].values) - et  # Effective red_begin (cycle end) time
    timex = np.concatenate([[0], np.repeat(cum_cycle, 2)[0:-1]])  # The x-axis for light

    cum_cycle0 = np.concatenate([[0], cum_cycle[0:-1]])  # The end time of last cycle
    y_red_begin = light_data['red_begin'] - et - cum_cycle0
    y_green_begin = light_data['green_begin'] + et - cum_cycle0
    y_red_begin = np.repeat(y_red_begin, 2)
    y_green_begin = np.repeat(y_green_begin, 2)
    y_red_begin[y_red_begin < 0] = np.nan
    y_green_begin[y_green_begin < 0] = np.nan

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(timex, y_red_begin, 'r')
    ax.plot(timex, y_green_begin, 'g')
    ax.fill_between(timex, y_green_begin, y_red_begin, color='g', alpha=0.2)
    ax.margins(x=0)
    ax.set_ybound(0, np.nanmax(y_red_begin) * 1.1)

    # Trajectory part
    t_x = loop_data['t_start'].values + L / v
    t_y = []
    for t in t_x:
        t_y.append(t - cum_cycle0[np.sum(t > cum_cycle0) - 1])
    plt.plot(t_x, t_y, 'k.', markersize=2)
    t_y = np.array(t_y)
    POG = (t_y>=np.nanmax(y_green_begin)).sum()/t_y.shape[0]

    ax.set_xlabel('Time in simulation (s)')
    ax.set_ylabel('Time in cycle (s)')
    return fig, ax, POG


def plot_trajectory(file, light_results, interval, ped_results, direction='WE'):
    cmap = plt.cm.jet_r(np.linspace(0, 14, 60) / 14)
    columns = ["id", "x", "y", "lane", "speed"]
    if direction == 'WE':
        lane_set = {'gneE14_0', 'gneE14_1', '-gneE17_0', '-gneE17_1', 'gneE18_0', 'gneE18_1', '-gneE19_0', '-gneE19_1'}
        ybound = [-350, 550]
    else:
        lane_set = {'-gneE14_0', '-gneE14_1', 'gneE17_0', 'gneE17_1', '-gneE18_0', '-gneE18_1', 'gneE19_0', 'gneE19_1'}
        ybound = [-550, 350]

    root = ET.parse(file)
    times = root.findall('.//timestep')
    data = [[v.get(column) for column in columns] + [t.get('time')] for t in times for v in t]
    data = pd.DataFrame(data=data, columns=columns + ['time'])
    data = data.loc[data.lane.isin(lane_set), :]

    data.x = data.x.astype(np.float32)
    data.time = data.time.astype(np.float32)
    data.speed = data.speed.astype(np.float32)
    data.sort_values(by=['id', 'time'], inplace=True)
    data.set_index('id', inplace=True)

    data1 = data.loc[(data.time >= interval[0]) & (data.time <= interval[1]), :]
    id_set = data1.index.unique()
    if direction == 'EW':
        data1.loc[:, 'x'] = - data1.loc[:, 'x']
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot trajectory
    for i in id_set:
        data2 = data1.loc[i, :].values.reshape([-1, 5])
        for ii in range(data2.shape[0] - 1):
            ax.plot([data2[ii, 4], data2[ii + 1, 4]], [data2[ii, 0], data2[ii + 1, 0]],
                    color=cmap[min(int(np.round(data2[ii+1, 3] / 14 * 60)), 59)], lw=0.5)


    # Plot Traffic light
    if direction == 'WE':
        light_pos = {'T01': -250, 'T02': 103, 'T03': 450}
    else:
        light_pos = {'T01': 250, 'T02': -103, 'T03': -450}
    light_color = {'T01': {0: 'r', 1: 'r', 2: 'g', 3: 'gold'},
                   'T02': {0: 'g', 1: 'gold', 2: 'r', 3: 'r'},
                   'T03': {0: 'r', 1: 'r', 2: 'g', 3: 'gold'}}
    for light, pos in light_pos.items():
        light_result = light_results[light]
        for t in range(interval[0], interval[1]):
            ax.plot([t, t+1], [pos, pos], color=light_color[light][light_result[t]], lw=3)

    # Plot pedestrian requst
    for ped in ped_results:
        if interval[0] < ped[0] < interval[1]:
            ax.plot(ped[0], light_pos['T02'], 'x', color='black', ms=8)

    ax.margins(x=0, y=0)
    sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(vmin=0, vmax=14))
    cb = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
    cb.set_label('Speed (m/s)')
    ax.set_ybound(ybound[0], ybound[1])
    return fig, ax


#%%
def plot_trajectory2(ax, file, light_results, interval, ped_results, direction='WE'):
    cmap = plt.cm.jet_r(np.linspace(0, 14, 60) / 14)
    columns = ["id", "x", "y", "lane", "speed"]
    if direction == 'WE':
        lane_set = {'gneE14_0', 'gneE14_1', '-gneE17_0', '-gneE17_1', 'gneE18_0', 'gneE18_1', '-gneE19_0', '-gneE19_1'}
        ybound = [-550, 350]
    else:
        lane_set = {'-gneE14_0', '-gneE14_1', 'gneE17_0', 'gneE17_1', '-gneE18_0', '-gneE18_1', 'gneE19_0', 'gneE19_1'}
        ybound = [-550, 350]

    root = ET.parse(file)
    times = root.findall('.//timestep')
    data = [[v.get(column) for column in columns] + [t.get('time')] for t in times for v in t]
    data = pd.DataFrame(data=data, columns=columns + ['time'])
    data = data.loc[data.lane.isin(lane_set), :]

    data.x = data.x.astype(np.float32)
    data.time = data.time.astype(np.float32)
    data.speed = data.speed.astype(np.float32)
    data.sort_values(by=['id', 'time'], inplace=True)
    data.set_index('id', inplace=True)

    data1 = data.loc[(data.time >= interval[0]) & (data.time <= interval[1]), :]
    id_set = data1.index.unique()
    # if direction == 'EW':
    data1.loc[:, 'x'] = - data1.loc[:, 'x']
    # fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot trajectory
    for i in id_set:
        data2 = data1.loc[i, :].values.reshape([-1, 5])
        for ii in range(data2.shape[0] - 1):
            ax.plot([data2[ii, 4], data2[ii + 1, 4]], [data2[ii, 0], data2[ii + 1, 0]],
                    color=cmap[min(int(np.round(data2[ii+1, 3] / 14 * 60)), 59)], lw=0.3)


    # # Plot Traffic light
    # if direction == 'WE':
    #     light_pos = {'T01': -250, 'T02': 103, 'T03': 450}
    # else:
    light_pos = {'T01': 250, 'T02': -99, 'T03': -450}
    light_color = {'T01': {0: 'r', 1: 'r', 2: 'g', 3: 'gold'},
                   'T02': {0: 'g', 1: 'gold', 2: 'r', 3: 'gold'},
                   'T03': {0: 'r', 1: 'r', 2: 'g', 3: 'gold'}}
    light_style = {'T01': {0: '-', 1: '-', 2: '-', 3: '-'},
                   'T02': {0: '-', 1: '-', 2: '-', 3: '-'},
                   'T03': {0: '-', 1: '-', 2: '-', 3: '-'}}
    for light, pos in light_pos.items():
        light_result = light_results[light]
        for t in range(interval[0], interval[1]):
            ax.plot([t, t+1], [pos, pos], color=light_color[light][light_result[t]],
                    lw=3, ls=light_style[light][light_result[t]],
                    dash_joinstyle='miter',dash_capstyle='butt')

    # Plot pedestrian requst
    for ped in ped_results:
        if interval[0] < ped[0] < interval[1]:
            ax.plot(ped[0], light_pos['T02'], 'x', color='black', ms=8)
            ax.axvline(ped[0], color='gray', linewidth=0.5)

    ax.margins(x=0, y=0)
    sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(vmin=0, vmax=14))
    # cb = plt.colorbar(sm, ax=ax, aspect=40, pad=0.02)
    # cb.set_label('Speed (m/s)')
    ax.set_ybound(ybound[0], ybound[1])
    return sm

#%%
def train_forecast_model(loop_results, traj_file):
    loop_data_EW = pd.concat([loop_result_to_dataframe(loop_results['-LP18_0']),
                              loop_result_to_dataframe(loop_results['-LP18_1'])])
    loop_data_EW.set_index('vID', inplace=True)
    loop_data_EW = loop_data_EW[~loop_data_EW.index.duplicated(keep='first')]

    loop_data_WE = pd.concat([loop_result_to_dataframe(loop_results['-LP17_0']),
                              loop_result_to_dataframe(loop_results['-LP17_1'])])
    loop_data_WE.set_index('vID', inplace=True)
    loop_data_WE = loop_data_WE[~loop_data_WE.index.duplicated(keep='first')]
    loop_data = {'WE':loop_data_WE, 'EW':loop_data_EW}

    light_cycle = {'WE':70, 'EW':80}
    models = {'WE':None, 'EW':None}
    columns = ["id", "lane", "pos", "speed"]

    # Process data for trajectory
    root = ET.parse(traj_file)
    times = root.findall('.//timestep')
    for direction in light_cycle.keys():
        data = [[v.get(column) for column in columns] + [t.get('time')] for t in times for v in t]
        data = pd.DataFrame(data=data, columns=columns + ['time'])
        # Select vehicles in the upstream
        data = data.loc[data.lane.isin(lane_sets[direction]), :]
        data.pos = data.pos.astype(np.float32)
        data.speed = data.speed.astype(np.float32)
        data.time = data.time.astype(np.float32)
        data.time = data.time.astype(np.int32)

        # Count how many vehicles in front of the lane
        data = data.sort_values(['time','lane'])
        data['n_ahead'] = data.groupby(['time', 'lane'])['pos'].transform(lambda x: np.argsort(-x).argsort())

        # Select vehicles that will pass the corridor
        data = data.loc[data.id.apply(lambda x: x[0:2]).isin(id_sets[direction]), :]
        data = data.groupby("id").sample(n=3, random_state=1)


        data.set_index('id', inplace=True)
        data['time_loop'] = loop_data[direction].t_start
        data['light'] = data.time % light_cycle[direction]
        data['travel_time'] = data.time_loop - data.time
        data['lane_number'] = data.lane.map(lane_sets[direction])
        data = data.sample(frac=1)

        # RF forecast
        X = data.loc[:, ['lane_number', 'light', 'pos', 'speed', 'n_ahead']].values
        Y = data.loc[:, 'travel_time'].values
        regr = RandomForestRegressor(random_state=0)
        regr.fit(X, Y)
        y_predict = regr.predict(X)
        print(mean_squared_error(Y, y_predict))
        models[direction] = regr
    return models

    # n = 2800
    # X = data.loc[:, ['lane_number', 'light', 'pos','speed','n_ahead']].values
    # Y = data.loc[:, 'travel_time'].values
    # regr = RandomForestRegressor(random_state=1)
    # regr.fit(X[0:n, :], Y[0:n])
    # y_predict = regr.predict(X[n:, :])
    # from sklearn.metrics import r2_score
    # print(mean_squared_error(Y[n:], y_predict))
    # print(r2_score(Y[n:], y_predict))
    # plt.figure()
    # plt.plot(Y[n:], y_predict, '+')

    # y_predict = regr.predict(X[0:n, :])
    # plt.figure()
    # plt.plot(Y[0:n], y_predict, '+')
    # mean_absolute_percentage_error(Y[n:], y_predict)

